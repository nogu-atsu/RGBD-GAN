# just for visualizatio

import os
import sys
import re
import json

import glob
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

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


def make_image(gen, prior, stage=6.5, n_images=20, name="", ignore_white=False, n_views=100, origin=np.zeros(3),
               range_phi=0.4, range_theta=0.2):
    print(stage)
    xp = gen.xp
    point_clouds = []
    for i in tqdm(range(0, n_images)):
        # z = Variable(xp.asarray(gen.make_hidden(n_images)))
        z = xp.asarray(gen.make_hidden(1))[:, None]
        z = Variable(xp.broadcast_to(z, (1, n_views, *z.shape[2:]))).reshape(-1, *z.shape[2:])

        theta = prior.sample(n_views)
        random_camera_matrices = xp.array(get_camera_matries(theta))
        theta = xp.array(np.concatenate([np.cos(theta[:, :3]), np.sin(theta[:, :3]), theta[:, 3:]], axis=1))

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            if config.generator_architecture == "deepvoxels":
                # z2 = xp.asarray(gen.make_hidden(n_images))
                z2 = xp.asarray(gen.make_hidden(1))[:, None]
                z2 = xp.broadcast_to(z2, (1, n_views, *z2.shape[2:])).reshape(-1, *z2.shape[2:])
                x = gen(z, stage, random_camera_matrices, z2=z2)
            elif config.generator_architecture in ["dcgan", "stylegan"]:
                x = gen(z, stage=stage, theta=theta)
            else:
                assert False
        loss_func_rotate = LossFuncRotate(xp)
        loss_func_rotate.init_params(xp, size=x.shape[2])
        point_cloud = loss_func_rotate.calc_real_pos(x, random_camera_matrices)
        point_cloud = chainer.backends.cuda.to_cpu(point_cloud)
        point_clouds.append(point_cloud)

    point_cloud = np.concatenate(point_clouds, axis=0)
    point_cloud = point_cloud.transpose(0, 2, 1).reshape(-1, 6)

    point_cloud[:, :3] = np.clip(point_cloud[:, :3] * 0.5 + 0.5, 0, 1)
    dist = dist_from_origin(point_cloud, origin, n_images, n_views,
                            range_phi=range_phi, range_theta=range_theta, ignore_white=ignore_white)[:, 1:-1, 1:-1]
    dist = dist.reshape(n_images, n_views, *dist.shape[1:])
    var = np.sum(dist ** 2 * (dist != -1), axis=1) / (np.sum(dist != -1, axis=1) + 1e-10) - \
          np.sum(dist * (dist != -1), axis=1) ** 2 / (np.sum(dist != -1, axis=1) + 1e-10) ** 2
    mean_var = np.sum(var, axis=(0, 1, 2)) / np.sum(var > 1e-7, axis=(0, 1, 2))
    return mean_var


def dist_from_origin(point_cloud, origin, batchsize=16, n_views=16, eps=1e-8, range_phi=0.4, range_theta=0.2,
                     ignore_white=False):
    """
    origin: where rays come from
    point_cloud: projected in real-world space
    """
    xyz = point_cloud[:, 3:] - origin
    rgb = point_cloud[:, :3]
    r = np.linalg.norm(xyz, axis=1, keepdims=True)
    phi = np.arcsin(xyz[:, 1:2] / (r + eps))
    theta = np.arctan2(xyz[:, :1], xyz[:, 2:])
    # sorted_phi = np.sort(phi.reshape(-1))
    # min_phi = sorted_phi[len(sorted_phi) // 20]
    # max_phi = sorted_phi[-(len(sorted_phi) // 20)]
    # sorted_theta = np.sort(theta.reshape(-1))
    # min_theta = sorted_theta[len(sorted_theta) // 20]
    # max_theta = sorted_theta[-(len(sorted_theta) // 20)]
    min_phi = -range_phi
    max_phi = range_phi
    min_theta = -range_theta
    max_theta = range_theta
    r = r.reshape(batchsize * n_views, -1, 1)
    phi = phi.reshape(batchsize * n_views, -1)
    theta = theta.reshape(batchsize * n_views, -1)
    rgb = rgb.reshape(batchsize * n_views, -1, 3)
    if ignore_white:
        r[np.where(np.sum(rgb, axis=1) >= 2.5)] = 10000
    cells = []
    for i in range(batchsize * n_views):
        args = np.argsort(r[i, :, 0])[::-1]
        r_ = r[i][args]
        phi_ = phi[i][args]
        theta_ = theta[i][args]
        rgb_ = rgb[i][args]
        if ignore_white:
            rgb_[np.where(r_ > 1000)] = -1
            r_[np.where(r_ > 1000)] = -1
        theta_num = int(range_theta * 80)
        phi_num = int(range_phi * 80)
        quantized_phi = np.floor((phi_ - min_phi) / (max_phi - min_phi) * phi_num).astype("int") + 1
        quantized_theta = np.floor((theta_ - min_theta) / (max_theta - min_theta) * theta_num).astype("int") + 1
        quantized_phi = np.clip(quantized_phi, 0, phi_num + 1)  # はみ出した奴をclip
        quantized_theta = np.clip(quantized_theta, 0, theta_num + 1)  # はみ出した奴をclip
        cell = np.full((phi_num + 2, theta_num + 2, 4), -1,
                       dtype="float")  # 4 channels for r&RGB, surrounding pixels are for はみ出した奴
        cell[quantized_phi, quantized_theta] = np.concatenate([r_, rgb_], axis=1)
        cells.append(cell)
    cells = np.array(cells)
    return cells


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
args = parser.parse_args()

models = {
    # "ffhq_dcgan": ["configs/reedbush/cvpr2020/ffhq_pggan_occlusion2.yml", 250000],
    # "ffhq_stylegan": ["configs/final_ffhq_stylegan_occlusion.yml", 250000],
    # "ffhq_deepvoxels": ["configs/final_ffhq_deepvoxels.yml", 65000],
    # "ffhq_hologan": ["configs/final_ffhq_hologan.yml", 65000],
    # "ffhq_rendernet": ["configs/final_ffhq_rendernet.yml", 62000],
    # "shapenet_car_dcgan": ["configs/final_shapenet_car_dcgan_occlusion.yml", 250000],
    # "shapenet_car_stylegan": ["configs/shapenet_car_stylegan_occlusion_low_geometric.yml", 250000],
    # "shapenet_car_deepvoxels": ["configs/final_shapenet_car_deepvoxels.yml", 62000],
    # "shapenet_car_hologan": ["configs/final_shapenet_car_hologan.yml", 62000],
    # "shapenet_car_rendernet": ["configs/final_shapenet_car_rendernet.yml", 62000],
    # "bedroom_stylegan": ["configs/final_bedroom_stylegan_occlusion_low_geometric.yml", 300000],
    # "car_stylegan": ["configs/car_stylegan_occlusion.yml", 300000]
    # "shapenet_car_accumulative_2": ["configs/final_shapenet_car_accumulative_2.yml", 65000],
    # "shapenet_car_accumulative_hologan_2": ["configs/final_shapenet_car_iclr_final_dcgan.yml", 65000],
    "shapenet_car_stylegan": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_stylegan_occlusion.yml", 260000],
    "shapenet_car_dcgan": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_dcgan_occlusion_low_geometric.yml", 260000],
    "shapenet_car_accumulative_hologan": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_accumulative_hologan.yml", 70000],
    "shapenet_car_accumulative": ["configs/iclr_camera_ready/final_shapenet_car_iclr_final_accumulative.yml", 70000],
}

chainer.backends.cuda.get_device_from_id(args.gpu).use()

log = ""
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

    origin = np.zeros(3) if "car" in model else np.array([0, 0, -0.5])
    range_phi = 3.14159 / 2 if "car" in model else 0.4
    range_theta = 3.14159 if "car" in model else 0.4
    ignore_white = "shapenet_car" in model
    results = make_image(generator, prior, stage, n_images=100,
                         name=model, ignore_white=ignore_white, origin=origin, range_phi=range_phi,
                         range_theta=range_theta)
    log += f"{model} {results[0]} {np.mean(results[1:])}\n"
    print(log)
