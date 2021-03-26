# this code is for
import os, sys, math, time, random
from pathlib import Path
import numpy as np
import argparse
import cv2
import chainer
from chainer import functions as F
from chainer import Variable
from PIL import Image
from tqdm import tqdm

from evaluation import load_inception_model
from evaluation import get_mean_cov, FID
import yaml
import utils.yaml_utils as yaml_utils
from train_rgbd import setup_generator, prepare_dataset, CameraParamPrior

from updater_deepvoxels import get_camera_matries


def make_image(gen, prior, stage=6.5, n_images=10000, batchsize=100, rgb=False):
    xp = gen.xp

    imgs = []
    for i in tqdm(range(0, n_images, batchsize)):
        z = xp.asarray(gen.make_hidden(batchsize))

        if rgb:
            theta = None
        else:
            theta = prior.sample(batchsize)
            random_camera_matrices = xp.array(get_camera_matries(theta), dtype="float32")
            theta = xp.array(np.concatenate([np.cos(theta[:, :3]), np.sin(theta[:, :3]), theta[:, 3:]], axis=1))

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            if config.generator_architecture == "deepvoxels":
                z2 = xp.asarray(gen.make_hidden(batchsize))
                x = gen(z, stage, random_camera_matrices, z2=z2)
            elif config.generator_architecture in ["dcgan", "stylegan"]:
                x = gen(z, stage=stage, theta=theta)
            else:
                assert False
        x = chainer.backends.cuda.to_cpu(x.array)[:, :3]
        imgs.append(x)

    return np.concatenate(imgs, axis=0)


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def resize_image(img, size):
    img_ = []
    for im in img:
        img_.append(cv2.resize(im.transpose(1, 2, 0).astype("uint8"), (size, size)))
    return np.array(img_).transpose(0, 3, 1, 2).astype("float32")


def get_features(model, ims, batch_size=100):
    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))
    print('Batch size:', batch_size)
    print('Total number of images:', n)
    print('Total number of batches:', n_batches)
    ys = np.empty((n, 2048), dtype=np.float32)
    for i in range(n_batches):
        print('Running batch', i + 1, '/', n_batches, '...')
        batch_start = (i * batch_size)
        batch_end = min((i + 1) * batch_size, n)

        ims_batch = ims[batch_start:batch_end]
        ims_batch = chainer.backends.cuda.to_gpu(ims_batch)  # To GPU if using CuPy

        # Resize image to the shape expected by the inception module
        if (w, h) != (299, 299):
            ims_batch = F.resize_images(ims_batch, (299, 299))  # bilinear

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = model(ims_batch, get_feature=True)
        ys[batch_start:batch_end] = chainer.cuda.to_cpu(y.data)
    return ys


def distance(x, y):
    return np.sum(x ** 2, axis=1, keepdims=True) + np.sum(y ** 2, axis=1, keepdims=True).T - 2 * np.dot(x, y.T)


def mmd(real, fake, sigma=1):
    Mxx = distance(real, real)
    Mxy = distance(real, fake)
    Myy = distance(fake, fake)
    scale = Mxx.mean()
    Mxx = np.exp(-Mxx / (scale * 2 * sigma ** 2))
    Mxy = np.exp(-Mxy / (scale * 2 * sigma ** 2))
    Myy = np.exp(-Myy / (scale * 2 * sigma ** 2))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())
    return mmd


def mv(real, fake):
    # realとfakeの分散の比
    var_real = real.var(axis=0).mean()
    var_fake = fake.var(axis=0).mean()
    return var_fake / var_real


def calc_mmd(model, real_ims, fake_ims, plot=True, name="hoge", mean_var=True):
    real = get_features(model, real_ims)
    fake = get_features(model, fake_ims)
    # plot_density(real, fake, name=name)
    return mmd(real, fake), mv(real, fake)


def save_image(gen_image, name, size=(5, 5)):
    # fake_image: output of the generator
    b, c, h, w = gen_image.shape
    if gen_image.shape[0] < size[0] * size[1]:
        n = int(gen_image.shape[0] ** 0.5)
        size = (n, n)
    gen_image = gen_image[:size[0] * size[1]]
    gen_image = gen_image.reshape((size[0], size[1], 3, h, w))
    gen_image = gen_image.transpose(0, 3, 1, 4, 2)
    gen_image = gen_image.reshape((size[0] * h, size[1] * w, 3))

    gen_image = gen_image.astype("uint8")
    print(gen_image.shape, gen_image.dtype)
    Image.fromarray(gen_image).save(f"eval_results/{name}.jpg")


def gen_images(params, gen, num, tmp=1., truncate=False, batchsize=100):
    img = []
    for i in range(0, num, batchsize):
        img.append(params.random_forward(gen, tmp, truncate, batchsize))
    img = np.concatenate(img, axis=0)
    return img


def gen_interpolate_images(params, gen):
    img = params.interpoalte_forward(gen)
    return img


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--stat_dir_path', type=str, default='/home/gk75/k75008/data/inception_model/')
    parser.add_argument('--inception_model_path', type=str,
                        default='/data/unagi0/noguchi/inception_model/inception_model')
    parser.add_argument('--eval_size', type=int, default=10000)
    parser.add_argument('--resize', type=int, default=299)
    # parser.add_argument('--tmp', type=float, default=0.4)
    parser.add_argument('--data_path', type=str, default="/data/unagi0/noguchi/dataset/")
    parser.add_argument('--model_path', type=str, default="/lustre/gk75/k75008/data/")
    parser.add_argument('--target_set', type=str, default="train")  # train or test set is used for evaluation
    args = parser.parse_args()

    chainer.cuda.get_device_from_id(args.gpu).use()

    models = {
        "ffhq_dcgan": ["configs/20191107_pggan_ffhq2.yml", 250000],
        # "ffhq_stylegan_rgb": ["configs/final_ffhq_stylegan_rgb.yml", 280000],
        # "ffhq_dcgan": ["configs/final_ffhq_dcgan_occlusion.yml", 300000],
        # "ffhq_stylegan": ["configs/final_ffhq_stylegan_occlusion.yml", 300000],
        # "ffhq_deepvoxels": ["configs/final_ffhq_deepvoxels.yml", 75000],
        # "ffhq_hologan": ["configs/final_ffhq_hologan.yml", 75000],
        # "ffhq_rendernet": ["configs/final_ffhq_rendernet.yml", 75000],
        #
        # "shapenet_car_dcgan_rgb": ["configs/final_shapenet_car_dcgan_rgb.yml", 260000],
        # "shapenet_car_stylegan_rgb": ["configs/final_shapenet_car_stylegan_rgb.yml", 300000],
        # "shapenet_car_dcgan": ["configs/final_shapenet_car_dcgan_occlusion.yml", 300000],
        # "shapenet_car_stylegan": ["configs/shapenet_car_stylegan_occlusion_low_geometric.yml", 300000],
        # "shapenet_car_deepvoxels": ["configs/final_shapenet_car_deepvoxels.yml", 58000],
        # "shapenet_car_hologan": ["configs/final_shapenet_car_hologan.yml", 58000],
        # "shapenet_car_rendernet": ["configs/final_shapenet_car_rendernet.yml", 75000],
        # "shapenet_car_accumulative": ["configs/final_shapenet_car_accumulative.yml", 75000],
        # "shapenet_car_accumulative_2": ["configs/final_shapenet_car_accumulative_2.yml", 75000],
        # "shapenet_car_accumulative_hologan": ["configs/final_shapenet_car_accumulative_hologan.yml", 75000],
        # "shapenet_car_accumulative_hologan_2": ["configs/final_shapenet_car_accumulative_hologan_2.yml", 75000],
    }

    np.random.seed(1234)

    inception_model = load_inception_model(args.inception_model_path)
    print("loaded inception model")

    log = ""
    for model, (config_path, iteration) in models.items():
        print(model)

        config = yaml_utils.Config(yaml.load(open(config_path)))
        generator = setup_generator(config).to_gpu()

        dataset = prepare_dataset(config)
        print("loaded dataset")
        # sample images randomly
        real_img = dataset[np.random.choice(len(dataset), args.eval_size, replace=False)]

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

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            eval_size = args.eval_size
            fake_img = make_image(generator, prior, stage=stage, n_images=args.eval_size, batchsize=100, rgb=config.rgb)

        fake_img = fake_img * 127.5 + 127.5

        if fake_img.shape[2] == 64:
            real_img = F.average_pooling_2d(real_img, 2, 2, 0).array

        # fid
        fake_mean, fake_cov = get_mean_cov(inception_model, fake_img, batch_size=100)
        real_mean, real_cov = get_mean_cov(inception_model, real_img, batch_size=100)
        fid = FID(real_mean, real_cov, fake_mean, fake_cov)
        log += f"{model}: FID={fid}\n"

        print(log)
