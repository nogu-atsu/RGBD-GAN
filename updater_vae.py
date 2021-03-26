#!/usr/bin/env python3
import chainer
import chainer.distributions as D
import chainer.functions as F
from chainer import Variable
import numpy as np
from config import get_lr_scale_factor
from scipy.stats import truncnorm

import chainer.computational_graph as c

from common.loss_functions import loss_func_dcgan_dis, loss_func_dcgan_gen, loss_l2, LossFuncRotate, SmoothDepth
from common.utils.copy_param import soft_copy_param
from common.utils.pggan import downsize_real


def loss_func_dsgan(x, z, theta, tau=10):
    if x.shape[1] == 4:
        x = x[:, :3]
    loss_ds_1 = F.batch_l2_norm_squared(x[::2] - x[1::2]) / (F.batch_l2_norm_squared(z[::2] - z[1::2]) + 1e-8)
    loss_ds_2 = F.batch_l2_norm_squared(x[::2] - x[1::2]) / (F.absolute(theta[::2] - theta[1::2]) + 1e-8) / 1000
    xp = chainer.cuda.get_array_module(x.array)
    loss_ds_1 = F.minimum(F.sqrt(loss_ds_1), xp.full_like(loss_ds_1.array, tau))
    loss_ds_2 = F.minimum(F.sqrt(loss_ds_2), xp.full_like(loss_ds_2.array, tau))
    print(loss_ds_1.array.mean(), loss_ds_2.array.mean())
    return -F.mean(loss_ds_1) - F.mean(loss_ds_2)


class VAEUpdater(chainer.training.updaters.StandardUpdater):
    def __init__(self, models, config, **kwargs):
        if len(models) == 2:
            models = models + [None]
        self.gen, self.enc, self.smoothed_gen = models

        # Stage manager
        self.config = config

        # Parse kwargs for updater
        # self.use_cleargrads = kwargs.pop('use_cleargrads')
        self.smoothing = kwargs.pop('smoothing')
        self.lambda_gp = kwargs.pop('lambda_gp')

        self.total_gpu = kwargs.pop('total_gpu')

        self.style_mixing_rate = kwargs.pop('style_mixing_rate')

        self.loss_func_rotate = LossFuncRotate(self.gen.xp)
        self.loss_smooth_depth = SmoothDepth(self.gen.xp)
        self.stage_interval = list(map(int, self.config.stage_interval.split(",")))
        super(VAEUpdater, self).__init__(**kwargs)

    def get_z_fake_data(self, batch_size):
        xp = self.gen.xp
        return xp.asarray(self.gen.make_hidden(batch_size))

    @property
    def stage(self):
        return self.get_stage()

    def get_stage(self):
        for i, interval in enumerate(self.stage_interval):
            if self.iteration + 1 <= interval:
                return i - 1 + (self.iteration - self.stage_interval[i - 1]) / (interval - self.stage_interval[i - 1])
        # return 6.5
        # return min(self.iteration / self.config.stage_interval+6, self.config.max_stage - 1e-8)

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

    def update_core(self):
        xp = self.gen.xp

        self.gen.cleargrads()
        self.enc.cleargrads()

        opt_g = self.get_optimizer('gen')
        opt_e = self.get_optimizer('enc')

        # z: latent | x: data | y: dis output
        # *_real/*_fake/*_pertubed: Variable
        # *_data: just data (xp array)

        stage = self.stage  # Need to retrive the value since next statement may change state (at the stage boundary)
        batch = self.get_iterator('main').next()
        batch_size = len(batch)

        # lr_scale = get_lr_scale_factor(self.total_gpu, stage)

        x_real_data = self.get_x_real_data(batch, batch_size)

        if isinstance(chainer.global_config.dtype, chainer._Mixed16):
            x_real_data = x_real_data.astype("float16")

        x_real = Variable(x_real_data)

        z = self.enc(x_real)
        theta = z[:, -1]

        loc, scale = F.split_axis(z[:, :-1], 2, 1)
        scale = F.softplus(scale - 5)
        # scale = F.minimum(scale, xp.full_like(scale.array, 0.05))
        q_z = D.Normal(loc, scale)
        p_z = D.Normal(xp.zeros_like(loc.array), xp.ones_like(scale.array))
        loss_kl = F.mean(chainer.kl_divergence(q_z, p_z)) * 0.3
        chainer.report({'loss_kl': loss_kl}, opt_g.target)
        loss_kl.backward()
        del loss_kl

        z_sample = q_z.sample()  # sample z

        z_sample = z_sample / F.sqrt(F.sum(z_sample * z_sample, axis=1, keepdims=True) / z.shape[0] + 1e-8)
        x_fake = self.gen(z_sample, stage, theta)

        x_real = downsize_real(x_real, stage)
        image_size = x_real.shape[2]
        loss_l1 = F.mean_squared_error(x_fake[:, :3], x_real)
        chainer.report({'loss_l1': loss_l1}, opt_g.target)
        loss_l1.backward()
        del loss_l1

        loss_theta = F.mean(F.relu(theta - 1) + F.relu(-1 - theta))  # regularize theta near 0
        loss_theta += F.relu(0.25 - (F.mean(theta ** 2) - F.mean(theta) ** 2))
        chainer.report({'loss_theta': loss_theta}, opt_g.target)
        loss_theta.backward()
        del loss_theta
        if self.iteration > self.config.start_rotation:
            noise = xp.random.uniform(0, 1, size=theta.shape[0], dtype="float32")
            theta2 = xp.maximum(theta.array - 0.5, -xp.ones_like(theta.array) * self.config.angle_range) * noise + \
                     xp.minimum(theta.array + 0.5, xp.ones_like(theta.array) * self.config.angle_range) * (1 - noise)

            x_fake2 = self.gen(z_sample, stage, theta2)

            loss_rotate, warped_zp = self.loss_func_rotate(x_fake, theta.array, x_fake2, theta2)
            loss_rotate = loss_rotate * 0.2 + F.mean(F.relu(1 - x_fake[:, -1])) * 0.6  # make depth larger than 1

            chainer.report({'loss_rotate': loss_rotate}, opt_g.target)
            loss_rotate.backward()
            del loss_rotate

        opt_g.update()
        opt_e.update()

        chainer.reporter.report({'stage': stage})
        chainer.reporter.report({'batch_size': batch_size})
        chainer.reporter.report({'image_size': image_size})
