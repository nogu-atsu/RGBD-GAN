# just for visualizatio

import os
import sys
import re
import json

import glob
import numpy as np
from PIL import Image
import cv2

import chainer
from chainer import Variable
from chainer import functions as F

import yaml
from net import Discriminator, StyleGenerator, MappingNetwork

# from chainer_profutil import create_marked_profile_optimizer

import utils.yaml_utils as yaml_utils
from updater_deepvoxels import get_camera_matries
from common.loss_functions import LossFuncRotate
from train_rgbd import setup_generator, CameraParamPrior


def make_image(gen, prior, stage=6.5, n_images=20, name=""):
    print(stage)
    xp = gen.xp

    z = Variable(xp.asarray(gen.make_hidden(n_images)))
    # z = xp.asarray(gen.make_hidden(1))[:, None]
    # z = Variable(xp.broadcast_to(z, (1, n_images, *z.shape[2:]))).reshape(-1, *z.shape[2:])

    theta = prior.sample(n_images)
    random_camera_matrices = xp.array(get_camera_matries(theta))
    theta = xp.array(np.concatenate([np.cos(theta[:, :3]), np.sin(theta[:, :3]), theta[:, 3:]], axis=1))

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        if config.generator_architecture == "deepvoxels":
            z2 = xp.asarray(gen.make_hidden(n_images))
            # z2 = xp.asarray(gen.make_hidden(1))[:, None]
            # z2 = xp.broadcast_to(z2, (1, n_images, *z2.shape[2:])).reshape(-1, *z2.shape[2:])
            x = gen(z, stage, random_camera_matrices, z2=z2)
        elif config.generator_architecture in ["dcgan", "stylegan"]:
            x = gen(z, stage=stage, theta=theta)
            print(x.shape)
        else:
            assert False
    loss_func_rotate = LossFuncRotate(xp)
    loss_func_rotate.init_params(xp, size=x.shape[2])
    point_cloud = loss_func_rotate.calc_real_pos(x, random_camera_matrices)

    point_cloud = chainer.backends.cuda.to_cpu(point_cloud)

    np.save(f"point_cloud_{name}.npy", point_cloud)
    np.save(f"x_{name}.npy", chainer.backends.cuda.to_cpu(x.array))


def convert_batch_images(x, rows, cols):
    rgbd = False
    if x.shape[1] == 4:
        rgbd = True
        depth = np.tile(x[:, -1:], (1, 3, 1, 1))
        x = x[:, :-1]
    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    if rgbd:
        depth = np.asarray(np.clip(1 / (depth) * 128, 0.0, 255.0), dtype=np.uint8)
        depth = depth.reshape((rows, cols, 3, H, W))
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
args = parser.parse_args()

models = {
    # "ffhq_dcgan": ["configs/reedbush/cvpr2020/ffhq_pggan_occlusion2.yml", 250000],
    # "ffhq_stylegan": ["configs/final_ffhq_stylegan_occlusion.yml", 300000],
    # "ffhq_deepvoxels": ["configs/final_ffhq_deepvoxels.yml", 65000],
    # "ffhq_hologan": ["configs/final_ffhq_hologan.yml", 62000],
    # "ffhq_rendernet": ["configs/final_ffhq_rendernet.yml", 62000],
    # "shapenet_car_dcgan": ["configs/final_shapenet_car_dcgan_occlusion.yml", 300000],
    "shapenet_car_stylegan_good": ["configs/iclr_camera_ready/final_shapenet_car_stylegan_occlusion.yml", 250000],
    # "shapenet_car_deepvoxels": ["configs/final_shapenet_car_deepvoxels.yml", 62000],
    # "shapenet_car_hologan": ["configs/final_shapenet_car_hologan.yml", 62000],
    # "shapenet_car_rendernet": ["configs/final_shapenet_car_rendernet.yml", 62000],
    # "bedroom_stylegan": ["configs/final_bedroom_stylegan_occlusion_low_geometric.yml", 300000],
    # "car_stylegan": ["configs/car_stylegan_occlusion.yml", 300000]
    # "shapenet_car_accumulative_2": ["configs/final_shapenet_car_accumulative_2.yml", 65000],
}

chainer.backends.cuda.get_device_from_id(args.gpu).use()

for model, (config_path, iteration) in models.items():
    print(model)

    config = yaml_utils.Config(yaml.load(open(config_path)))
    generator = setup_generator(config).to_gpu()

    prior = CameraParamPrior(config)

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

    stage = min(config.max_stage - 1e-8, stage)

    make_image(generator, prior, stage, n_images=30, name=model)
