#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys

import chainer
import chainer.cuda
import cupy
import numpy as np
from PIL import Image
from chainer import Variable
from chainer import training
from chainer.datasets import TransformDataset
from chainer.training import extension
from chainer.training import extensions
from scipy import io
from tqdm import tqdm

try:
    import chainermn
    from mpi4py import MPI

    mpi_is_master = False
    mpi_available = True
except:  # pylint:disable=bare-except
    mpi_is_master = True
    mpi_available = False

import yaml

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

from common.utils.record import record_setting
from common.utils.save_images import convert_batch_images

import utils.yaml_utils as yaml_utils


def sample_generate_light(gen, dst, config, rows=8, cols=8, z=None, seed=0, subdir='preview'):
    @chainer.training.make_extension()
    def make_image(trainer):
        nonlocal rows, cols, z

        np.random.seed(seed)
        n_images = cols
        xp = gen.xp
        if z is None:
            if config.rgb:
                z = xp.asarray(gen.make_hidden(rows * cols))
            else:
                z = xp.asarray(gen.make_hidden(n_images))
                z = xp.tile(z[:, None], (1, rows) + (1,) * (z.ndim - 1)).reshape(rows * cols, *z.shape[1:])
        else:
            z = z[:n_images * rows]

        if config.rgb:
            theta = None
        else:
            theta = np.zeros((rows * cols, 6))
            theta[:, 1] = np.tile(np.linspace(-config.test_y_rotate, config.test_y_rotate, rows), cols)
            theta = theta.astype("float32")

            from updater_deepvoxels import get_camera_matries
            random_camera_matrices = xp.array(get_camera_matries(theta), dtype="float32")
            theta = xp.array(np.concatenate([np.cos(theta[:, :3]), np.sin(theta[:, :3]),
                                             theta[:, 3:]], axis=1))

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            if config.generator_architecture == "deepvoxels":
                z2 = xp.asarray(gen.make_hidden(n_images))
                z2 = xp.tile(z2[:, None], (1, rows) + (1,) * (z2.ndim - 1)).reshape(rows * cols, *z.shape[1:])
                x = gen(z, trainer.updater.stage, random_camera_matrices, z2=z2, theta=theta)
            elif config.generator_architecture in ["dcgan", "stylegan"]:
                x = gen(z, stage=trainer.updater.stage, theta=theta)
            else:
                assert False
        x = chainer.cuda.to_cpu(x.data)

        np.random.seed()

        x = convert_batch_images(x, rows, cols)

        preview_dir = '{}/{}'.format(dst, subdir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        preview_path = preview_dir + '/image_latest.png'
        Image.fromarray(x).save(preview_path)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration // 2000 * 2000)
        Image.fromarray(x).save(preview_path)

    return make_image


def make_iterator_func(dataset, batch_size):
    return chainer.iterators.MultithreadIterator(dataset, batch_size=batch_size, repeat=True, shuffle=None,
                                                 n_threads=config.dataset_worker_num)


def batch_generate_func(gen, mapping, trainer):
    def generate(n_images):
        xp = gen.xp
        z = Variable(xp.asarray(mapping.make_hidden(n_images)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(mapping(z), stage=trainer.updater.stage)
        x = chainer.cuda.to_cpu(x.data)
        return x

    return generate


class RunningHelper(object):

    def __init__(self, use_mpi, config):
        self.use_mpi = use_mpi
        self.config = config
        self.nvprof = config.nvprof

        # Setup
        if self.use_mpi:
            if not mpi_available:
                raise RuntimeError('ChainerMN required for MPI but cannot be imported. Abort.')
            comm = chainermn.create_communicator(config.comm_name)
            if comm.mpi_comm.rank == 0:
                print('==========================================')
                print('Num process (COMM_WORLD): {}'.format(MPI.COMM_WORLD.Get_size()))
                print('Communcator name: {}'.format(config.comm_name))
                print('==========================================')
            fleet_size = MPI.COMM_WORLD.Get_size()
            device = comm.intra_rank
        else:
            fleet_size = 1
            comm = None
            device = config.gpu

        self.fleet_size, self.comm, self.device = fleet_size, comm, device

        self.is_master = is_master = not self.use_mpi or (self.use_mpi and comm.rank == 0)

        # Early works
        if is_master:
            record_setting(config.out)

        # Show effective hps
        # effective_hps = {
        #     'is_master': self.is_master,
        #     'dynamic_batch_size': self.dynamic_batch_size
        # }
        # self.print_log('Effective hps: {}'.format(effective_hps))

    @property
    def keep_smoothed_gen(self):
        return self.config.keep_smoothed_gen and self.is_master

    @property
    def use_cleargrads(self):
        # 1. Chainer 2/3 does not support clear_grads when running with MPI, so use zero_grads instead
        # 2. zero_grads on chainer >= 5.0.0 has a critical bug when running with MPI
        return True

    @property
    def stage_interval(self):
        return self.config.stage_interval // self.fleet_size

    @property
    def dynamic_batch_size(self):
        fleet_size = self.fleet_size
        return [int(_) for _ in self.config.dynamic_batch_size.split(',')]

    def print_log(self, msg):
        print('[Device {}] {}'.format(self.device, msg))

    def check_hps_consistency(self):
        assert self.config.max_stage % 2 == 1
        # assert 4 * (2 ** ((config.max_stage - 1) // 2)) == config.image_size
        # assert 2 ** int(np.floor(np.log2(config.image_size))) == config.image_size
        assert len(self.dynamic_batch_size) >= self.config.max_stage

    def make_optimizer(self, model, alpha, beta1, beta2):
        self.print_log('Use Adam Optimizer with alpah = {}, beta1 = {}, beta2 = {}'.format(alpha, beta1, beta2))
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        if self.use_mpi:
            self.print_log('Use Optimizer with MPI')
            optimizer = chainermn.create_multi_node_optimizer(optimizer, self.comm)
        # if self.nvprof:
        # optimizer = create_marked_profile_optimizer(optimizer, sync=True, sync_level=2)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(5))
        return optimizer

    # def make_dataset(self, stage_int):
    #     if self.is_master:
    #         size = 4 * (2 ** ((stage_int + 1) // 2))
    #         _dataset = BaseDataset(
    #             json.load(open(self.config.dataset_config, 'r')),
    #             '%dx%d' % (size, size),
    #             [["resize", {"probability": 1, "width": size, "height": size, "resample_filter": "ANTIALIAS"}]]
    #         )
    #         self.print_log('Add (master) dataset for size {}'.format(size))
    #     else:
    #         _dataset = None
    #         self.print_log('Add (slave) dataset')
    #
    #     if self.use_mpi:
    #         _dataset = chainermn.scatter_dataset(_dataset, self.comm)
    #
    #     return _dataset


def crop_square(img):
    w, h = img.size
    size = min(w, h)
    return img.crop((
        (w - size) // 2, (h - size) // 2, (w + size) // 2, (h + size) // 2
    ))


def make_dataset(dataset_path, image_path):
    if os.path.exists(f"{dataset_path}/images.npy"):
        return np.load(f"{dataset_path}/images.npy").astype("float32")
    else:
        paths = glob.glob(image_path)
        imgs = []
        for p in tqdm(paths):
            img = Image.open(p)
            img = np.array(img).transpose(2, 0, 1)
            imgs.append(img)
        imgs = np.array(imgs, dtype="uint8")
        np.save(f"{dataset_path}/images.npy", imgs)
        return imgs.astype("float32")


def make_ffhq_dataset(reedbush=False):
    if reedbush:
        if os.path.exists("/lustre/gk75/k75008/data/cvpr2020/ffhq128.npy"):
            return np.load("/lustre/gk75/k75008/data/cvpr2020/ffhq128.npy").astype("float32")
        else:
            assert False
    else:
        if os.path.exists("/data/unagi0/noguchi/dataset/ffhq-dataset/ffhq128.npy"):
            return np.load("/data/unagi0/noguchi/dataset/ffhq-dataset/ffhq128.npy").astype("float32")
        else:
            paths = glob.glob("/data/unagi0/noguchi/dataset/ffhq-dataset/thumbnails128x128/*.png")
            imgs = []
            for p in tqdm(paths):
                img = Image.open(p)
                img = np.array(img).transpose(2, 0, 1)
                imgs.append(img)
            imgs = np.array(imgs, dtype="uint8")
            np.save("/data/unagi0/noguchi/dataset/ffhq-dataset/ffhq128.npy", imgs)
            return imgs.astype("float32")


def make_car_dataset(reedbush=False, ratio=0.6, ):
    if reedbush:
        if os.path.exists("/lustre/gk75/k75008/data/cvpr2020/cars_align.npy"):
            return np.load("/lustre/gk75/k75008/data/cvpr2020/cars_align.npy").astype("float32")
    else:
        if os.path.exists(f"/data/unagi0/noguchi/dataset/stanford_cars/cars_align_{ratio}_area.npy"):
            return np.load(f"/data/unagi0/noguchi/dataset/stanford_cars/cars_align_{ratio}_area.npy").astype("float32")
        else:
            dir_name = ["cars_test", "cars_train"]
            for dn in dir_name:
                paths = glob.glob(f"/data/unagi0/noguchi/dataset/stanford_cars/{dn}/*.jpg")
                anno = io.loadmat(f"/data/unagi0/noguchi/dataset/stanford_cars/devkit/{dn}_annos.mat")
                x1 = np.array([a[0][0] for a in anno["annotations"]['bbox_x1'][0]])
                x2 = np.array([a[0][0] for a in anno["annotations"]['bbox_x2'][0]])
                y1 = np.array([a[0][0] for a in anno["annotations"]['bbox_y1'][0]])
                y2 = np.array([a[0][0] for a in anno["annotations"]['bbox_y2'][0]])

                imgs = []
                for i in tqdm(range(len(paths))):
                    img = Image.open(
                        f"/data/unagi0/noguchi/dataset/stanford_cars/{dn}/{i + 1:0>5d}.jpg").convert('RGB')
                    h, w = img.size
                    img = np.array(img)
                    height = (int(y2[i] - y1[i]) * int(x2[i] - x1[i]) / ratio) ** 0.5
                    x_center = (x1[i] + x2[i]) / 2
                    y_center = (y1[i] + y2[i]) / 2
                    x_1 = int(x_center - height / 2)
                    x_2 = int(x_center + height / 2)
                    y_1 = int(y_center - height / 2)
                    y_2 = int(y_center + height / 2)

                    pad = ((max(0, -y_1), max(y_2 - w, 0)), (max(0, -x_1), max(x_2 - h, 0)), (0, 0))
                    img = np.pad(img[max(0, y_1):min(w, y_2), max(0, x_1):min(h, x_2)], pad, "edge")

                    img = Image.fromarray(img)
                    if min(img.size) < 128:
                        continue
                    img = img.resize((128, 128), Image.LANCZOS)
                    img = np.array(img).transpose(2, 0, 1)
                    imgs.append(img)
            imgs = np.array(imgs, dtype="uint8")
            np.save(f"/data/unagi0/noguchi/dataset/stanford_cars/cars_align_{ratio}_area.npy", imgs)
            return imgs.astype("float32")


def make_bedroom_dataset(reedbush=False):
    if reedbush:
        if os.path.exists("/lustre/gk75/k75008/data/cvpr2020/bedroom128.npy"):
            return np.load("/lustre/gk75/k75008/data/cvpr2020/bedroom128.npy").astype("float32")
    else:
        if os.path.exists("/data/unagi0/noguchi/dataset/lsun/bedroom128.npy"):
            return np.load("/data/unagi0/noguchi/dataset/lsun/bedroom128.npy").astype("float32")
        else:
            paths = glob.glob("/data/unagi0/noguchi/dataset/lsun/train128/*.jpg")
            imgs = []
            for p in tqdm(paths[:100000]):
                img = Image.open(p)
                img = np.array(img).transpose(2, 0, 1)
                imgs.append(img)
            imgs = np.array(imgs, dtype="uint8")
            np.save("/data/unagi0/noguchi/dataset/lsun/bedroom128.npy", imgs)
            return imgs.astype("float32")


def make_chair_dataset(reedbush=False):
    if reedbush:
        if os.path.exists("/lustre/gk75/k75008/data/cvpr2020/chair128.npy"):
            return np.load("/lustre/gk75/k75008/data/cvpr2020/chair128.npy").astype("float32")
    else:
        if os.path.exists("/data/unagi0/noguchi/dataset/rendered_shapenet/chair/chair128.npy"):
            return np.load("/data/unagi0/noguchi/dataset/rendered_shapenet/chair/chair128.npy").astype("float32")
        else:
            paths = sorted(glob.glob("/data/unagi0/noguchi/dataset/rendered_shapenet/chair/*.png"))
            imgs = []
            for p in tqdm(paths[:400000]):
                if "png0001.png" in p:
                    continue
                img = Image.open(p)
                img.load()
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
                img = np.array(img).transpose(2, 0, 1)
                imgs.append(img)
            imgs = np.array(imgs, dtype="uint8")
            np.save("/data/unagi0/noguchi/dataset/rendered_shapenet/chair/chair128.npy", imgs)
            return imgs.astype("float32")


def make_shapenet_car_dataset(reedbush=False, iclr_final=False):
    iclr_final = "_iclr_final" * iclr_final
    if reedbush:
        if os.path.exists("/lustre/gk75/k75008/data/cvpr2020/shapenet_car128.npy"):
            return np.load("/lustre/gk75/k75008/data/cvpr2020/shapenet_car128.npy").astype("float32")
        else:
            assert False
    else:
        if os.path.exists(f"/data/unagi0/noguchi/dataset/rendered_shapenet/car{iclr_final}/car128.npy"):
            return np.load(f"/data/unagi0/noguchi/dataset/rendered_shapenet/car{iclr_final}/car128.npy").astype(
                "float32")
        else:
            paths = sorted(glob.glob(f"/data/unagi0/noguchi/dataset/rendered_shapenet/car{iclr_final}/*.png"))
            imgs = []
            for p in tqdm(paths[:400000]):
                if "png0001.png" in p:
                    continue
                img = Image.open(p)
                img.load()
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
                img = np.array(img).transpose(2, 0, 1)
                imgs.append(img)
            imgs = np.array(imgs, dtype="uint8")
            np.save(f"/data/unagi0/noguchi/dataset/rendered_shapenet/car{iclr_final}/car128.npy", imgs)
            return imgs.astype("float32")


def make_shapenet_airplane_dataset():
    if os.path.exists("/data/unagi0/noguchi/dataset/rendered_shapenet/airplane/airplane128.npy"):
        return np.load("/data/unagi0/noguchi/dataset/rendered_shapenet/airplane/airplane128.npy").astype("float32")
    else:
        paths = sorted(glob.glob("/data/unagi0/noguchi/dataset/rendered_shapenet/airplane/*.png"))
        imgs = []
        for p in tqdm(paths[:400000]):
            if "png0001.png" in p:
                continue
            img = Image.open(p)
            img.load()
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
            img = np.array(img).transpose(2, 0, 1)
            imgs.append(img)
        imgs = np.array(imgs, dtype="uint8")
        np.save("/data/unagi0/noguchi/dataset/rendered_shapenet/airplane/airplane128.npy", imgs)
        return imgs.astype("float32")


def prepare_dataset(config):
    dataset = make_dataset(config.dataset_path, config.image_path)
    # if config.dataset == "ffhq":
    #     dataset = make_ffhq_dataset(config.reedbush)
    # elif config.dataset == "bedroom":
    #     dataset = make_bedroom_dataset(config.reedbush)
    # elif config.dataset in ["car", "car_0.6", "car_0.35"]:
    #     if config.dataset == "car":
    #         ratio = 0.6
    #     else:
    #         ratio = float(config.dataset.split("_")[1])
    #     dataset = make_car_dataset(config.reedbush, ratio=ratio)
    # elif config.dataset == "chair":
    #     dataset = make_chair_dataset()
    # elif config.dataset in ["shapenet_car", "shapenet_car_iclr_final"]:
    #     dataset = make_shapenet_car_dataset(config.reedbush, iclr_final=("iclr_final" in config.dataset))
    # elif config.dataset == "airplane":
    #     dataset = make_shapenet_airplane_dataset()
    # else:
    #     assert False

    return dataset


#
# def update_camera_matrices(mat, axis1, axis2, theta):
#     """
#     camera parameters update for get_camera_matrices function
#     :param mat:
#     :param axis1: int 0~2
#     :param axis2: int 0~2
#     :param theta: np array of rotation degree
#     :return: camara matrices of minibatch
#     """
#     rot = np.zeros_like(mat)
#     rot[:, range(4), range(4)] = 1
#     rot[:, axis1, axis1] = np.cos(theta)
#     rot[:, axis1, axis2] = -np.sin(theta)
#     rot[:, axis2, axis1] = np.sin(theta)
#     rot[:, axis2, axis2] = np.cos(theta)
#     mat = np.matmul(rot, mat)
#     return mat
#
#
# def get_camera_matries(thetas, order=(1,0,2)):
#     """
#     generate camera matrices from thetas
#     :param thetas: batchsize x 6, [x, y, z_rotation, x, y, z_translation]
#     :return:
#     """
#     mat = np.zeros((len(thetas), 4, 4), dtype="float32")
#     mat[:, range(4), range(4)] = [1, 1, -1, 1]
#     mat[:, 2, 3] = 1
#
#     for i in order:  # y, x, z_rotation
#         mat = update_camera_matrices(mat, (i + 1) % 3, (i + 2) % 3, thetas[:, i])
#
#     mat[:, :3, 3] = mat[:, :3, 3] + thetas[:, 3:]
#
#     return mat


class CameraParamPrior:
    def __init__(self, config):
        self.rotation_range = np.array([config.x_rotate, config.y_rotate, config.z_rotate])
        self.camera_param_range = np.array([config.x_rotate, config.y_rotate, config.z_rotate,
                                            config.x_translate, config.y_translate, config.z_translate])
        self.uniform = config.uniform_distribution

    def sample(self, batch_size):
        thetas = np.random.uniform(-1, 1, size=(batch_size // 2, 6))
        eps = np.random.uniform(0, 0.5, size=(batch_size // 2, 6))
        sign = np.random.choice(2, size=(batch_size // 2, 3)) * 2 - 1
        if self.uniform:
            eps[:, :3] = eps[:, :3] * sign * \
                         np.clip(1 / (self.rotation_range + 1e-8), 0, 1)  # limit angle difference
        else:
            eps[:, :3] = eps[:, :3] * (sign * (self.rotation_range == 3.1415) +
                                       np.abs(sign) * (self.rotation_range != 3.1415)) * \
                         np.clip(1 / (self.rotation_range + 1e-8), 0, 1)  # limit angle difference
        thetas2 = -eps * np.sign(thetas) + thetas
        if self.uniform:
            thetas2 = thetas2 * (-1 <= thetas2) * (thetas2 <= 1) + (-2 - thetas2) * (thetas2 < -1) + \
                      (2 - thetas2) * (thetas2 > 1)
        thetas = np.concatenate([thetas, thetas2], axis=0)

        thetas = thetas * self.camera_param_range[None]
        return thetas.astype("float32")


def setup_generator(config):
    rgbd = False if config.rgb else True
    if config.generator_architecture == "stylegan":
        from net import StyleGANGenerator
        generator = StyleGANGenerator(config.ch, enable_blur=config.enable_blur, rgbd=rgbd,
                                      rotate_conv_input=config.rotate_conv_input, use_encoder=config.bigan,
                                      use_occupancy_net=config.use_occupancy_net_loss,
                                      initial_depth=config.initial_depth)
    elif config.generator_architecture == "dcgan":
        from net import DCGANGenerator
        generator = DCGANGenerator(config.ch, enable_blur=config.enable_blur, rgbd=rgbd, use_encoder=config.bigan,
                                   use_occupancy_net=config.use_occupancy_net_loss,
                                   initial_depth=config.initial_depth)
    elif config.generator_architecture == "deepvoxels":
        from deepvoxels_generator import Generator
        if config.rendernet_projection:
            occlusion_type = "rendernet"
        elif config.occlusion_type:
            occlusion_type = config.occlusion_type
        else:
            occlusion_type = "deepvoxels"
        generator = Generator(config.ch, occlusion_type=occlusion_type,
                              background_generator=config.background_generator,
                              config=config)
    else:
        assert False, f"{config.generator_architecture} is not supported"
    return generator


def setup_discriminator(config):
    from net import Discriminator, BigBiGANDiscriminator
    num_z = 1 if config.generator_architecture == "dcgan" else 2
    if hasattr(config, "bigan") and config.bigan:
        discriminator = BigBiGANDiscriminator(config.ch, config.ch * num_z, enable_blur=config.enable_blur,
                                              sn=config.sn, res=config.res_dis)
    else:
        discriminator = Discriminator(ch=config.ch, enable_blur=config.enable_blur, sn=config.sn,
                                      res=config.res_dis)
    return discriminator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/ffhq_progressive.yml")
    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    config.gpu = args.gpu

    print(config.stage_interval)

    print("dtype = ", chainer.global_config.dtype)

    # FLAGS(sys.argv)

    running_helper = RunningHelper(config.use_mpi, config)
    global mpi_is_master
    mpi_is_master = running_helper.is_master
    # Check stage / image size / dynamic batch size / data consistency.
    # running_helper.check_hps_consistency()

    # Setup Models
    generator = setup_generator(config)

    discriminator = setup_discriminator(config)
    if config.sn:
        print("Spectral normalization discriminator")

    if running_helper.keep_smoothed_gen:
        smoothed_generator = setup_generator(config)

    models = [generator, discriminator]
    model_names = ['Generator', 'Discriminator']
    if running_helper.keep_smoothed_gen:
        models.append(smoothed_generator)
        model_names.append('SmoothedGenerator')

    if running_helper.device > -1:
        chainer.cuda.get_device_from_id(running_helper.device).use()

        generator.to_gpu()
        discriminator.to_gpu()
        if config.generator_architecture == "deepvoxels":
            generator.mapping.to_gpu()

    dataset = prepare_dataset(config)

    train = TransformDataset(dataset, lambda x: x / 127.5 - 1)
    # train_iter = chainer.iterators.MultiprocessIterator(train, config.batchsize, n_processes=4)
    train_iter = chainer.iterators.SerialIterator(train, config.batchsize)

    prior = CameraParamPrior(config)
    # stage_manager = StageManager(
    #     stage_interval=running_helper.stage_interval,
    #     dynamic_batch_size=running_helper.dynamic_batch_size,
    #     make_dataset_func=running_helper.make_dataset,
    #     make_iterator_func=make_iterator_func,
    #     debug_start_instance=config.debug_start_instance)

    # if running_helper.is_master:
    #    chainer.global_config.debug = True

    if config.generator_architecture == "stylegan":
        optimizer = {
            "map": running_helper.make_optimizer(generator.mapping, config.adam_alpha_g / 100, config.adam_beta1,
                                                 config.adam_beta2),
            "gen": running_helper.make_optimizer(generator.gen, config.adam_alpha_g, config.adam_beta1,
                                                 config.adam_beta2),
            "dis": running_helper.make_optimizer(discriminator, config.adam_alpha_d, config.adam_beta1,
                                                 config.adam_beta2)
        }
        if not config.rgb:
            generator.gen.l1.c.W.update_rule.hyperparam.alpha = config.adam_alpha_g / 100
            generator.gen.l2.c.W.update_rule.hyperparam.alpha = config.adam_alpha_g / 100
            generator.gen.l1.c.b.update_rule.hyperparam.alpha = config.adam_alpha_g / 100
            generator.gen.l2.c.b.update_rule.hyperparam.alpha = config.adam_alpha_g / 100
    elif config.generator_architecture == "dcgan":
        optimizer = {
            "gen": running_helper.make_optimizer(generator, config.adam_alpha_g, config.adam_beta1,
                                                 config.adam_beta2),
            "dis": running_helper.make_optimizer(discriminator, config.adam_alpha_d, config.adam_beta1,
                                                 config.adam_beta2)
        }
    elif config.generator_architecture == "deepvoxels":
        optimizer = {
            "map": running_helper.make_optimizer(generator.mapping, config.adam_alpha_g / 100, config.adam_beta1,
                                                 config.adam_beta2),
            "gen": running_helper.make_optimizer(generator, config.adam_alpha_g, config.adam_beta1,
                                                 config.adam_beta2),
            "dis": running_helper.make_optimizer(discriminator, config.adam_alpha_d, config.adam_beta1,
                                                 config.adam_beta2)
        }

    # for opt in optimizer.values():
    #     opt.add_hook(chainer.optimizer_hooks.GradientClipping(0.1))

    updater_args = {
        "models": models,
        "optimizer": optimizer,
        "iterator": train_iter,
        'config': config,
        'lambda_gp': config.lambda_gp,
        'smoothing': config.smoothing,
        'total_gpu': running_helper.fleet_size,
        "prior": prior
    }
    if config.generator_architecture == "deepvoxels":
        from updater_deepvoxels import DeepVoxelsUpdater as Updater
    elif config.rgb:
        from updater import RGBUpdater as Updater
    else:
        from updater import RGBDUpdater as Updater
    updater = Updater(**updater_args)

    if config.nvprof or config.enable_cuda_profiling:
        config.iteration = 10

    trainer = training.Trainer(updater, (config.iteration, 'iteration'), out=config.out)

    # Set up extensions
    if running_helper.is_master:
        for model, model_name in zip(models, model_names):
            trainer.extend(
                extensions.snapshot_object(model, model_name + '_{.updater.iteration}.npz'),
                trigger=(config.snapshot_interval, 'iteration'))
        if config.generator_architecture == "deepvoxels":
            trainer.extend(
                extensions.snapshot_object(generator.mapping, 'Map_{.updater.iteration}.npz'),
                trigger=(config.snapshot_interval, 'iteration'))

        trainer.extend(
            extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
            trigger=(config.snapshot_interval, 'iteration'))

        trainer.extend(
            extensions.ProgressBar(update_interval=1 if config.nvprof or config.enable_cuda_profiling else 10))

        trainer.extend(
            sample_generate_light(generator, config.out, config, rows=8, cols=8),
            trigger=(config.evaluation_sample_interval, 'iteration'),
            priority=extension.PRIORITY_WRITER)

        if running_helper.keep_smoothed_gen:
            trainer.extend(
                sample_generate_light(smoothed_generator, config.out, config, rows=8, cols=8,
                                      subdir='preview_smoothed'),
                trigger=(config.evaluation_sample_interval, 'iteration'),
                priority=extension.PRIORITY_WRITER)

        report_keys = [
            'iteration', 'elapsed_time', 'stage', 'batch_size', 'image_size', 'gen/loss_adv', 'dis/loss_adv',
            'gen/loss_recon', 'dis/loss_gp', 'gen/loss_rotate', 'gen/loss_occupancy'
        ]
        if config.fid_interval > 0:
            assert False, "FID is not supported for debug."
            # report_keys += 'FID'
            # fidapi = FIDAPI(config.fid_clfs_type,
            #                 config.fid_clfs_path,
            #                 gpu=running_helper.device,
            #                 load_real_stat=config.fid_real_stat)
            # trainer.extend(
            #     fid_extension(fidapi,
            #                   batch_generate_func(generator, mapping, trainer),
            #                   seed=config.seed,
            #                   report_key='FID'
            #                   ),
            #     trigger=(config.fid_interval, 'iteration')
            # )
            # if running_helper.keep_smoothed_gen:
            #     report_keys += 'S_FID'
            #     trainer.extend(
            #         fid_extension(fidapi,
            #                       batch_generate_func(smoothed_generator, smoothed_mapping, trainer),
            #                       seed=config.seed,
            #                       report_key='S_FID'
            #                       ),
            #         trigger=(config.fid_interval, 'iteration')
            #     )

        trainer.extend(extensions.LogReport(keys=report_keys, trigger=(config.display_interval, 'iteration')))
        trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))

    # Recover if possible
    if config.get_model_from_interation != '':
        resume_iteration_str = config.get_model_from_interation
        print('Resume from {}'.format(resume_iteration_str))
        for model, model_name in zip(models, model_names):
            chainer.serializers.load_npz(
                config.out + '/' + model_name + '_%s.npz' % resume_iteration_str,
                model, strict=False)
        chainer.serializers.load_npz(
            config.out + '/' + 'snapshot_iter_%s.npz' % resume_iteration_str,
            trainer, strict=False)

    elif config.auto_resume:
        print("Auto Resume")
        candidates = []
        auto_resume_dir = config.auto_resume_dir if config.auto_resume_dir != '' else config.out
        for fname in [f for f in os.listdir(auto_resume_dir) if f.startswith('Generator_') and f.endswith('.npz')]:
            fname = re.sub(r'^Generator_', '', fname)
            fname = re.sub('\.npz$', '', fname)
            fname_int = None
            try:
                fname_int = int(fname)
            except ValueError:
                pass
            if fname_int is not None:
                all_model_exist = True
                for m in model_names:
                    if not os.path.exists(auto_resume_dir + '/' + m + '_' + fname + '.npz'):
                        all_model_exist = False

                if not os.path.exists(auto_resume_dir + '/' + ('snapshot_iter_%s.npz' % fname)):
                    all_model_exist = False

                if all_model_exist:
                    candidates.append(fname)

        # print(candidates)
        candidates.sort(key=lambda _: int(_), reverse=True)
        if len(candidates) > 0:
            resume_iteration_str = candidates[0]
        else:
            resume_iteration_str = None
        if resume_iteration_str is not None:
            print('Automatic resuming: use iteration %s' % resume_iteration_str)
            for model, model_name in zip(models, model_names):
                chainer.serializers.load_npz(
                    auto_resume_dir + '/' + model_name + '_%s.npz' % resume_iteration_str,
                    model, strict=False)
            chainer.serializers.load_npz(
                auto_resume_dir + '/' + 'snapshot_iter_%s.npz' % resume_iteration_str,
                trainer, strict=False)
            if config.generator_architecture == "deepvoxels":
                chainer.serializers.load_npz(
                    auto_resume_dir + '/' + 'Map_%s.npz' % resume_iteration_str,
                    generator.mapping, strict=False)

    # Run the training
    if config.enable_cuda_profiling:
        with cupy.cuda.profile():
            trainer.run()
    else:
        # with chainer.using_config('debug', True):
        trainer.run()

    for model, model_name in zip(models, model_names):
        chainer.serializers.save_npz(config.out + '/' + model_name + '_latest.npz', model)


if __name__ == '__main__':
    main()
