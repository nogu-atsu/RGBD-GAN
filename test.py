# just for visualization

import os
import sys

import chainer
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from chainer import Variable

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

import utils.yaml_utils as yaml_utils


def make_image(gen, stage=6.5, n_images=100):
    print(stage)
    xp = gen.xp
    z = xp.asarray(gen.make_hidden(1))
    z = Variable(xp.broadcast_to(z, (n_images, *z.shape[1:])))

    theta = np.zeros((n_images, 6))
    theta[:, 1] = np.linspace(-3.1415, 3.1415, n_images)
    # theta[:, 0] = np.concatenate([np.linspace(-0.2, 0.2, n_images // 2), np.linspace(0.2, -0.2, n_images // 2)])
    theta = theta.astype("float32")
    from updater_deepvoxels import get_camera_matries
    random_camera_matrices = xp.array(get_camera_matries(theta), dtype="float32")
    theta = xp.array(np.concatenate([np.cos(theta[:, :3]), np.sin(theta[:, :3]),
                                     theta[:, 3:]], axis=1))

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        if config.generator_architecture == "deepvoxels":
            z2 = xp.asarray(gen.make_hidden(1))
            z2 = xp.broadcast_to(z2, (n_images, *z2.shape[1:]))
            x = gen(z, stage, random_camera_matrices, z2=z2, theta=theta)
        elif config.generator_architecture in ["dcgan", "stylegan"]:
            x = gen(z, stage=stage, theta=theta)
        else:
            assert False
    x = chainer.backends.cuda.to_cpu(x.array)
    # Define the codec and create VideoWriter object
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    size = x.shape[2]
    writer = cv2.VideoWriter(f'{config.out}/rotate_depth.mp4', fmt, 30, (size * 2, size))  # ライター作成

    for i in range(10):
        for im in x:
            rgb = im[:3].transpose(1, 2, 0) * 127.5 + 127.5
            depth = im[-1:].transpose(1, 2, 0)
            H, W, _ = depth.shape
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
            depth = depth.reshape((H, W, 3))

            # depth = np.tile(1 / im[-1:].transpose(1, 2, 0) * 200 - 150, (1, 1, 3))
            im = np.concatenate([rgb, depth], axis=1)
            im = np.clip(im[:, :, ::-1], 0, 255).astype("uint8")
            writer.write(im)  # 画像を1フレーム分として書き込み

    writer.release()  # ファイルを閉じる

    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    size = x.shape[2]
    writer = cv2.VideoWriter(f'{config.out}/rotate.mp4', fmt, 30, (size, size))  # ライター作成

    for i in range(10):
        for im in x:
            rgb = im[:3].transpose(1, 2, 0) * 127.5 + 127.5
            im = np.clip(rgb[:, :, ::-1], 0, 255).astype("uint8")
            writer.write(im)  # 画像を1フレーム分として書き込み

    writer.release()  # ファイルを閉じる

    # return make_image


def make_image_shape_interpolationn(gen, stage=6.5, n_images=20):
    print(stage)
    batchsize = 100
    xp = gen.xp
    z = chainer.backends.cuda.to_cpu(gen.make_hidden(1000 // n_images + 1))
    z = np.swapaxes(np.linspace(z[:-1], z[1:], n_images), 1, 0).reshape(-1, *z.shape[1:])
    z = xp.array(z) * 0.8

    theta = np.zeros((batchsize, 6))
    # theta[:, 1] = np.linspace(-3.1415, 3.1415, n_images)
    # theta[:, 0] = np.concatenate([np.linspace(-0.2, 0.2, n_images // 2), np.linspace(0.2, -0.2, n_images // 2)])
    theta = theta.astype("float32")
    from updater_deepvoxels import get_camera_matries
    random_camera_matrices = xp.array(get_camera_matries(theta), dtype="float32")
    theta = xp.array(np.concatenate([np.cos(theta[:, :3]), np.sin(theta[:, :3]),
                                     theta[:, 3:]], axis=1))
    xs = []
    if config.generator_architecture == "deepvoxels":
        z2 = chainer.backends.cuda.to_cpu(gen.make_hidden(1000 // n_images + 1))
        z2 = np.swapaxes(np.linspace(z2[:-1], z2[1:], n_images), 0, 1).reshape(-1, *z2.shape[1:])
        z2 = xp.array(z2) * 0.8

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i in range(0, 1000, batchsize):
            if config.generator_architecture == "deepvoxels":
                print(z[i:i + batchsize].shape, z2[i:i + batchsize].shape)
                x = gen(z[i:i + batchsize], stage, random_camera_matrices, z2=z2[i:i + batchsize])
            elif config.generator_architecture in ["dcgan", "stylegan"]:
                x = gen(z[i:i + batchsize], stage=stage, theta=theta)
            else:
                assert False
            x = chainer.backends.cuda.to_cpu(x.array)
            xs.append(x)

    xs = np.concatenate(xs, axis=0)
    # Define the codec and create VideoWriter object
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    size = xs.shape[2]
    writer = cv2.VideoWriter(f'{config.out}/rotate.mp4', fmt, 30, (size * 2, size))  # ライター作成

    for im in xs:
        rgb = im[:3].transpose(1, 2, 0) * 127.5 + 127.5
        depth = np.tile(1 / im[-1:].transpose(1, 2, 0) * 200 - 150, (1, 1, 3))
        im = np.concatenate([rgb, depth], axis=1)
        im = np.clip(im[:, :, ::-1], 0, 255).astype("uint8")
        writer.write(im)  # 画像を1フレーム分として書き込み

    writer.release()  # ファイルを閉じる

    # return make_image


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
parser.add_argument("--iteration", type=int)
args = parser.parse_args()

config = yaml_utils.Config(yaml.load(open(args.config_path)))

from train_rgbd import setup_generator

chainer.backends.cuda.get_device_from_id(args.gpu).use()
generator = setup_generator(config).to_gpu()

if config.generator_architecture == "deepvoxels":
    generator.mapping.to_gpu()

resume_iteration_str = args.iteration

print('Resume from {}'.format(resume_iteration_str))
chainer.serializers.load_npz(config.out + '/Generator_%s.npz' % resume_iteration_str, generator, )
if config.generator_architecture == "deepvoxels":
    chainer.serializers.load_npz(config.out + '/Map_%s.npz' % resume_iteration_str, generator.mapping, )

stage_interval = list(map(int, config.stage_interval.split(",")))
iteration = int(resume_iteration_str)

if config.generator_architecture == "deepvoxels":
    stage = 8.5
else:
    stage = 0
    for i, interval in enumerate(stage_interval):
        if iteration <= interval:
            stage = i - 1 + (iteration - stage_interval[i - 1]) / (interval - stage_interval[i - 1])
            break
    else:
        stage = config.max_stage - 1e-8
# make_image_shape_interpolationn(generator, stage)
make_image(generator, stage)
