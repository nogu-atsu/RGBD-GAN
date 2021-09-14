#!/usr/bin/env python3
import chainer
import chainer.computational_graph as c
import chainer.functions as F
import numpy as np
from chainer import Variable

from common.loss_functions import loss_func_dcgan_dis, loss_func_dcgan_gen, loss_l2, LossFuncRotate, SmoothDepth
from common.utils.copy_param import soft_copy_param


def loss_func_dsgan(x, z, theta, tau=10):
    if x.shape[1] == 4:
        x = x[:, :3]
    loss_ds_1 = F.batch_l2_norm_squared(x[::2] - x[1::2]) / (F.batch_l2_norm_squared(z[::2] - z[1::2]) + 1e-8)
    loss_ds_2 = F.batch_l2_norm_squared(x[::2] - x[1::2]) / (F.absolute(theta[::2] - theta[1::2]) + 1e-8) / 1000
    xp = chainer.cuda.get_array_module(x.array)
    loss_ds_1 = F.minimum(F.sqrt(loss_ds_1), xp.full_like(loss_ds_1.array, tau))
    loss_ds_2 = F.minimum(F.sqrt(loss_ds_2), xp.full_like(loss_ds_2.array, tau))
    return -F.mean(loss_ds_1) - F.mean(loss_ds_2)


def downsize_real(x, size):
    w = x.shape[-1]
    scale = w // size
    return F.average_pooling_2d(x, scale, scale, 0)


def update_camera_matrices(mat, axis1, axis2, theta):
    """
    camera parameters update for get_camera_matrices function
    :param mat:
    :param axis1: int 0~2
    :param axis2: int 0~2
    :param theta: np array of rotation degree
    :return: camara matrices of minibatch
    """
    rot = np.zeros_like(mat)
    rot[:, range(4), range(4)] = 1
    rot[:, axis1, axis1] = np.cos(theta)
    rot[:, axis1, axis2] = -np.sin(theta)
    rot[:, axis2, axis1] = np.sin(theta)
    rot[:, axis2, axis2] = np.cos(theta)
    mat = np.matmul(rot, mat)
    return mat


def get_camera_matries(thetas, order=(0, 1, 2)):
    """
    generate camera matrices from thetas
    :param thetas: batchsize x 6, [x, y, z_rotation, x, y, z_translation]
    :return:
    """
    mat = np.zeros((len(thetas), 4, 4), dtype="float32")
    mat[:, range(4), range(4)] = [1, 1, -1, 1]
    mat[:, 2, 3] = 1

    for i in order:  # y, x, z_rotation
        mat = update_camera_matrices(mat, (i + 1) % 3, (i + 2) % 3, thetas[:, i])

    mat[:, :3, 3] = mat[:, :3, 3] + thetas[:, 3:]

    return mat


def calc_distance(est_theta, theta):
    # weak regularization to the distribution of estimated thetas
    dist = F.sum(est_theta ** 2, axis=1) + (theta ** 2).sum(axis=1).T - 2 * F.matmul(est_theta, theta, transb=True)

    return F.mean(F.min(dist, axis=0)) + F.mean(F.min(dist, axis=1))


IMG_SIZE = 64


class DeepVoxelsUpdater(chainer.training.updaters.StandardUpdater):
    def __init__(self, models, config, **kwargs):
        if len(models) == 2:
            models = models + [None, None]
        self.gen, self.dis, self.smoothed_gen, self.smoothed_map = models

        # Stage manager
        self.config = config

        self.smoothing = kwargs.pop('smoothing')
        self.lambda_gp = kwargs.pop('lambda_gp')

        self.total_gpu = kwargs.pop('total_gpu')
        self.prior = kwargs.pop("prior")

        lambda_geometric = self.config.lambda_geometric if self.config.lambda_geometric else 3  # 3 is default
        self.loss_func_rotate = LossFuncRotate(self.gen.xp, K=self.gen.projection.projection_intrinsic,
                                               lambda_geometric=lambda_geometric)
        self.loss_smooth_depth = SmoothDepth(self.gen.xp)
        self.stage_interval = list(map(int, self.config.stage_interval.split(",")))

        self.camera_param_range = np.array([config.x_rotate, config.y_rotate, config.z_rotate,
                                            config.x_translate, config.y_translate, config.z_translate])
        super(DeepVoxelsUpdater, self).__init__(**kwargs)

    @property
    def stage(self):
        return self.get_stage()

    def get_stage(self):
        return 8.5

    def get_x_real_data(self, batch, batch_size):
        xp = self.gen.xp
        x_real_data = []
        for i in range(batch_size):
            this_instance = batch[i]
            if isinstance(this_instance, tuple):
                this_instance = this_instance[0]  # It's (data, data_id), so take the first one.
            x_real_data.append(np.asarray(this_instance).astype("f"))
        x_real_data = xp.asarray(x_real_data)
        return x_real_data

    def get_z_fake_data(self, batch_size):
        xp = self.gen.xp
        return xp.asarray(self.gen.mapping.make_hidden(batch_size))

    def update_core(self):
        xp = self.gen.xp

        use_rotate = True if self.iteration > self.config.start_rotation else False
        self.gen.cleargrads()
        self.gen.mapping.cleargrads()
        self.dis.cleargrads()

        opt_g_m = self.get_optimizer('map')
        opt_g_g = self.get_optimizer('gen')
        opt_d = self.get_optimizer('dis')

        # z: latent | x: data | y: dis output
        # *_real/*_fake/*_pertubed: Variable
        # *_data: just data (xp array)

        stage = self.stage  # Need to retrive the value since next statement may change state (at the stage boundary)
        batch = self.get_iterator('main').next()
        batch_size = len(batch)

        # lr_scale = get_lr_scale_factor(self.total_gpu, stage)

        x_real_data = self.get_x_real_data(batch, batch_size)
        z_fake_data = xp.tile(self.get_z_fake_data(batch_size // 2), (2, 1, 1, 1, 1))  # repeat same z

        z_fake_data2 = xp.tile(self.get_z_fake_data(batch_size // 2), (2, 1, 1, 1, 1))  # repeat same z

        if isinstance(chainer.global_config.dtype, chainer._Mixed16):
            x_real_data = x_real_data.astype("float16")
            z_fake_data = z_fake_data.astype("float16")
            z_fake_data2 = z_fake_data.astype("float16")

        # TODO
        # theta->6 DOF
        if self.config.use_posterior:
            # theta is a combination of prior and posterior
            assert False, "not implemented yet"
        else:
            thetas = self.prior.sample(batch_size)

        # theta -> camera matrix
        random_camera_matrices = xp.array(get_camera_matries(thetas), dtype="float32")
        thetas = xp.array(np.concatenate([np.cos(thetas[:, :3]), np.sin(thetas[:, :3]),
                                          thetas[:, 3:]], axis=1))

        x_real = Variable(x_real_data)
        # Image.fromarray(convert_batch_images(x_real.data.get(), 4, 4)).save('no_downsized.png')
        x_real = downsize_real(x_real, IMG_SIZE)
        x_real = Variable(x_real.data)
        # Image.fromarray(convert_batch_images(x_real.data.get(), 4, 4)).save('downsized.png')
        image_size = x_real.shape[2]

        x_fake = self.gen(z_fake_data, stage, random_camera_matrices, z2=z_fake_data2, theta=thetas)
        y_fake = self.dis(x_fake[:, :3], stage=stage)
        loss_gen = loss_func_dcgan_gen(y_fake, self.config.focal_loss_gamma)  # * lr_scale

        chainer.report({'loss_adv': loss_gen}, self.gen)
        assert not xp.isnan(loss_gen.data)

        if use_rotate:
            if self.config.background_generator:
                loss_rotate_fore, _ = self.loss_func_rotate(x_fake[:batch_size // 2],
                                                            random_camera_matrices[:batch_size // 2],
                                                            x_fake[batch_size // 2:],
                                                            random_camera_matrices[batch_size // 2:],
                                                            max_depth=3)
                virtual_camera_matrices = random_camera_matrices.copy()
                virtual_camera_matrices[:, :3, 3] = 0
                loss_rotate_back, _ = self.loss_func_rotate(x_fake[:batch_size // 2],
                                                            virtual_camera_matrices[:batch_size // 2],
                                                            x_fake[batch_size // 2:],
                                                            virtual_camera_matrices[batch_size // 2:],
                                                            min_depth=3)

                loss_rotate = loss_rotate_fore + loss_rotate_back

            else:
                loss_rotate, _ = self.loss_func_rotate(x_fake[:batch_size // 2],
                                                       random_camera_matrices[:batch_size // 2],
                                                       x_fake[batch_size // 2:],
                                                       random_camera_matrices[batch_size // 2:])

            loss_rotate += F.mean(F.relu(self.config.depth_min - x_fake[:, -1]) ** 2) * \
                           self.config.lambda_depth  # make depth larger
            chainer.report({'loss_rotate': loss_rotate}, self.gen)
            assert not xp.isnan(loss_rotate.data)
            lambda_loss_rotate = self.config.lambda_loss_rotate if self.config.lambda_loss_rotatec else 0.3
            loss_gen = loss_gen + loss_rotate * lambda_loss_rotate

        if chainer.global_config.debug:
            g = c.build_computational_graph(loss_gen)
            with open('out_loss_gen', 'w') as o:
                o.write(g.dump())
        # assert not xp.isnan(loss_dsgan.data)
        loss_gen.backward()
        opt_g_m.update()
        opt_g_g.update()
        del loss_gen, y_fake, x_fake

        self.dis.cleargrads()
        # keep smoothed generator if instructed to do so.
        if self.smoothed_gen is not None:
            # layers_in_use = self.gen.get_layers_in_use(stage=stage)
            soft_copy_param(self.smoothed_gen, self.gen, 1.0 - self.smoothing)

        z_fake_data = self.get_z_fake_data(batch_size)
        z_fake_data2 = self.get_z_fake_data(batch_size)
        if isinstance(chainer.global_config.dtype, chainer._Mixed16):
            z_fake_data = z_fake_data.astype("float16")
            z_fake_data2 = z_fake_data.astype("float16")
        # with chainer.using_config('enable_backprop', False):
        x_fake = self.gen(z_fake_data, stage, random_camera_matrices, z2=z_fake_data2, theta=thetas)
        x_fake.unchain_backward()
        y_fake = self.dis(x_fake[:, :3], stage=stage)
        y_real = self.dis(x_real, stage=stage)
        loss_adv = loss_func_dcgan_dis(y_fake, y_real)

        if not self.dis.sn and self.lambda_gp > 0:
            x_perturbed = x_real
            y_perturbed = y_real
            # y_perturbed = self.dis(x_perturbed, stage=stage)
            grad_x_perturbed, = chainer.grad([y_perturbed], [x_perturbed], enable_double_backprop=True)
            grad_l2 = F.sqrt(F.sum(grad_x_perturbed ** 2, axis=(1, 2, 3)))
            loss_gp = self.lambda_gp * loss_l2(grad_l2, 0.0)
            chainer.report({'loss_gp': loss_gp}, self.dis)
        else:
            loss_gp = 0.
        loss_dis = (loss_adv + loss_gp)  # * lr_scale
        assert not xp.isnan(loss_dis.data)

        chainer.report({'loss_adv': loss_adv}, self.dis)

        loss_dis.backward()
        opt_d.update()

        chainer.reporter.report({'batch_size': batch_size})
        chainer.reporter.report({'image_size': image_size})
