# just for visualizatio

import os
import sys
import re
import json

import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import chainer
from chainer import Variable

import yaml
from net import Discriminator, StyleGenerator, MappingNetwork
from updater_deepvoxels import get_camera_matries

# from chainer_profutil import create_marked_profile_optimizer

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

import utils.yaml_utils as yaml_utils


def make_image(gen, stage=6.5, n_images=20, model_name="dcgan", iteration=0, angle_range=(1.0472, 0.3054)):
    print(stage)
    xp = gen.xp

    # azimuth
    interpolate_num = 8
    z = xp.asarray(gen.make_hidden(n_images))[:, None]
    z = Variable(xp.broadcast_to(z, (n_images, interpolate_num, *z.shape[2:]))).reshape(-1, *z.shape[2:])
    theta = np.zeros((interpolate_num, 6))
    theta[:, 1] = np.linspace(-angle_range[0], angle_range[0], interpolate_num)
    theta = np.tile(theta, (n_images, 1)).reshape(-1, 6)
    # theta[:, 0] = np.concatenate([np.linspace(-0.2, 0.2, n_images // 2), np.linspace(0.2, -0.2, n_images // 2)])
    theta = theta.astype("float32")
    from updater_deepvoxels import get_camera_matries
    random_camera_matrices = xp.array(get_camera_matries(theta), dtype="float32")
    theta = xp.array(np.concatenate([np.cos(theta[:, :3]), np.sin(theta[:, :3]),
                                     theta[:, 3:]], axis=1))

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        if config.generator_architecture == "deepvoxels":
            z2 = xp.asarray(gen.make_hidden(n_images))[:, None]
            z2 = xp.broadcast_to(z2, (n_images, interpolate_num, *z2.shape[2:])).reshape(-1, *z2.shape[2:])
            x = gen(z, stage, random_camera_matrices, z2=z2)
        elif config.generator_architecture in ["dcgan", "stylegan"]:
            x = gen(z, stage=stage, theta=theta)
            print(x.shape)
        else:
            assert False
    x = chainer.backends.cuda.to_cpu(x.array)

    x = convert_batch_images(x, n_images, interpolate_num)

    preview_dir = 'eval_results'
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)

    preview_path = preview_dir + f'/azimuth_{model_name}_{iteration}.png'
    Image.fromarray(x).save(preview_path)

    # elevation

    interpolate_num = 6
    z = xp.asarray(gen.make_hidden(n_images))[:, None]
    z = Variable(xp.broadcast_to(z, (n_images, interpolate_num, *z.shape[2:]))).reshape(-1, *z.shape[2:])

    theta = np.zeros((interpolate_num, 6))
    theta[:, 0] = np.linspace(-angle_range[1], angle_range[1], interpolate_num)
    theta = np.tile(theta, (n_images, 1)).reshape(-1, 6)
    # theta[:, 0] = np.concatenate([np.linspace(-0.2, 0.2, n_images // 2), np.linspace(0.2, -0.2, n_images // 2)])
    theta = theta.astype("float32")
    from updater_deepvoxels import get_camera_matries
    random_camera_matrices = xp.array(get_camera_matries(theta), dtype="float32")
    theta = xp.array(np.concatenate([np.cos(theta[:, :3]), np.sin(theta[:, :3]),
                                     theta[:, 3:]], axis=1))

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        if config.generator_architecture == "deepvoxels":
            z2 = xp.asarray(gen.make_hidden(n_images))[:, None]
            z2 = xp.broadcast_to(z2, (n_images, interpolate_num, *z2.shape[2:])).reshape(-1, *z2.shape[2:])
            x = gen(z, stage, random_camera_matrices, z2=z2)
        elif config.generator_architecture in ["dcgan", "stylegan"]:
            x = gen(z, stage=stage, theta=theta)
            print(x.shape)
        else:
            assert False
    x = chainer.backends.cuda.to_cpu(x.array)

    x = convert_batch_images(x, n_images, interpolate_num)

    preview_dir = 'eval_results'
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)

    preview_path = preview_dir + f'/elevation_{model_name}_{iteration}.png'
    Image.fromarray(x).save(preview_path)


def make_random_image(gen, prior, stage=6.5, n_images=100, model_name="dcgan", iteration=0, rgb=False):
    xp = gen.xp

    z = xp.asarray(gen.make_hidden(n_images)) * 0.9

    if rgb:
        theta = None
    else:
        theta = prior.sample(n_images)
        random_camera_matrices = xp.array(get_camera_matries(theta), dtype="float32")
        theta = xp.array(np.concatenate([np.cos(theta[:, :3]), np.sin(theta[:, :3]), theta[:, 3:]], axis=1))

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        if config.generator_architecture == "deepvoxels":
            z2 = xp.asarray(gen.make_hidden(n_images))
            x = gen(z, stage, random_camera_matrices, z2=z2)
        elif config.generator_architecture in ["dcgan", "stylegan"]:
            x = gen(z, stage=stage, theta=theta)
        else:
            assert False
    x = chainer.backends.cuda.to_cpu(x.array)[:, :3]
    x = x[:int(n_images ** 0.5) ** 2]
    x = convert_batch_images(x, int(n_images ** 0.5), int(n_images ** 0.5))

    preview_dir = 'eval_results'
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)

    preview_path = preview_dir + f'/random_{model_name}_{iteration}.png'
    Image.fromarray(x).save(preview_path)


def convert_batch_images(x, rows, cols):
    rgbd = False
    if x.shape[1] == 4:
        rgbd = True
        depth = x[:, -1:]
        x = x[:, :-1]
    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    if rgbd:
        depth = 1 / depth
        sorted_d = np.sort(depth.reshape(-1))
        min_d = sorted_d[len(sorted_d) // 50]
        max_d = sorted_d[-(len(sorted_d) // 50)]
        min_d = min_d * 1.4 - max_d * 0.4
        max_d = max_d * 1.4 - min_d * 0.4
        depth = (depth - min_d) / (max_d - min_d) * 255
        depth = np.asarray(np.clip(depth, 0.0, 255.0), dtype=np.uint8)
        cm = plt.get_cmap('viridis')  # 'jet')
        depth = cm(depth.reshape(-1))[:, :3] * 255
        depth = depth.reshape((rows, cols, H, W, 3)).transpose(0, 1, 4, 2, 3).astype("uint8")
        x = np.concatenate([x, depth], axis=1).reshape(rows * 2, cols, 3, H, W)
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((-1, cols * W, 3))
    return x


def batch_generate_func(gen, mapping, trainer):
    def generate(n_images):
        xp = gen.xp
        z = Variable(xp.asarray(mapping.make_hidden(n_images)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(mapping(z), stage=trainer.updater.stage)
        x = chainer.cuda.to_cpu(x.data)
        return x

    return generate


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", "-g", type=int, default=0)
parser.add_argument("--config_path", type=str, default="configs/ffhq_progressive.yml")
parser.add_argument("--random", action="store_true")
args = parser.parse_args()

from train_rgbd import setup_generator

models = {
    # "car_stylegan": ["configs/reedbush/cvpr2020/car_stylegan4.yml", 280000],
    # "car_dcgan": ["configs/cvpr2020/car_pggan_occlusion.yml", 280000],
    # "car_deepvoxels": ["configs/reedbush/cvpr2020/car_deepvoxels6.yml", 65000],
    # "car_hologan": ["configs/reedbush/cvpr2020/car_hologan2.yml", 65000],
    # "bedroom_deepvoxels5": ["configs/reedbush/cvpr2020/bedroom_deepvoxels5.yml", 65000],
    # "bedroom_deepvoxels6": ["configs/reedbush/cvpr2020/bedroom_deepvoxels6.yml", 65000],
    # "bedroom_stylegan": ["configs/reedbush/cvpr2020/bedroom_stylegan.yml", 200000],
    # "ffhq_dcgan": ["configs/reedbush/cvpr2020/ffhq_pggan_occlusion2.yml", 250000],
    # "ffhq_stylegan": ["configs/final_ffhq_stylegan_occlusion.yml", 300000],
    # "ffhq_deepvoxels": ["configs/final_ffhq_deepvoxels.yml", 75000],
    # "ffhq_hologan": ["configs/final_ffhq_hologan.yml", 75000],
    # "ffhq_rendernet": ["configs/final_ffhq_rendernet.yml", 75000],
    # "shapenet_car_dcgan": ["configs/final_shapenet_car_dcgan_occlusion.yml", 300000],
    # "shapenet_car_stylegan": ["configs/shapenet_car_stylegan_occlusion_low_geometric.yml", 300000],
    # "shapenet_car_deepvoxels": ["configs/final_shapenet_car_deepvoxels.yml", 58000],
    # "shapenet_car_hologan": ["configs/final_shapenet_car_hologan.yml", 58000],
    # "shapenet_car_rendernet": ["configs/final_shapenet_car_rendernet.yml", 75000],
    # "shapenet_car_accumulative": ["configs/final_shapenet_car_accumulative.yml", 75000],
    # "shapenet_car_accumulative_2": ["configs/final_shapenet_car_accumulative_2.yml", 75000],
    # "shapenet_car_accumulative_hologan": ["configs/final_shapenet_car_accumulative_hologan.yml", 75000],
    # "shapenet_car_accumulative_hologan_2": ["configs/final_shapenet_car_accumulative_hologan_2.yml", 75000],
    # 2020/02/13
    # "shapenet_car_iclr_final_stylegan": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_stylegan_occlusion.yml", 260000],
    # "shapenet_car_iclr_final_dcgan": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_dcgan_occlusion_low_geometric.yml", 260000],
    # "shapenet_car_iclr_final_accumulative_hologan": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_accumulative_hologan.yml", 70000],
    # "shapenet_car_iclr_final_accumulative": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_accumulative.yml", 70000],
    # "shapenet_car_iclr_final_stylegan_rgb": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_stylegan_rgb.yml", 260000],
    # "shapenet_car_iclr_final_dcgan_rgb": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_dcgan_rgb.yml", 260000],
    # 2020/02/14
    "car0.6_stylegan": ["configs/iclr_camera_ready/final_car0.6_stylegan_occlusion_initial_depth_0.8.yml", 300000],
    "car0.6_accumulative": ["configs/iclr_camera_ready/final_car0.6_iclr_final_accumulative.yml", 100000],
}

chainer.backends.cuda.get_device_from_id(args.gpu).use()

for model, (config_path, iteration) in models.items():
    print(model)

    config = yaml_utils.Config(yaml.load(open(config_path)))
    generator = setup_generator(config).to_gpu()

    if config.generator_architecture == "deepvoxels":
        generator.mapping.to_gpu()

    print('Resume from {}'.format(iteration))
    chainer.serializers.load_npz(config.out + '/Generator_%s.npz' % iteration, generator, )
    if config.generator_architecture == "deepvoxels":
        chainer.serializers.load_npz(config.out + '/Map_%s.npz' % iteration, generator.mapping, )

    stage_interval = list(map(int, config.stage_interval.split(",")))
    stage = 0
    for i, interval in enumerate(stage_interval):
        if iteration <= interval:
            stage = i - 1 + (iteration - stage_interval[i - 1]) / (interval - stage_interval[i - 1])
            break
    else:
        stage = config.max_stage - 1e-8
    stage = min(stage, config.max_stage - 1e-8)
    angle_range = (3.1415, 0.3054) if "car" in model else (1.0472, 0.3054)
    print(angle_range)
    if not args.random:
        make_image(generator, stage, model_name=model, iteration=iteration, angle_range=angle_range)
    else:
        stage_interval = list(map(int, config.stage_interval.split(",")))
        stage = 0
        for i, interval in enumerate(stage_interval):
            if iteration <= interval:
                stage = i - 1 + (iteration - stage_interval[i - 1]) / (interval - stage_interval[i - 1])
                break
        else:
            stage = config.max_stage - 1e-8

        stage = min(config.max_stage - 1e-8, stage)

        from train_rgbd import CameraParamPrior

        prior = CameraParamPrior(config)
        make_random_image(generator, prior, stage, n_images=100, model_name=model, iteration=iteration, rgb=config.rgb)
