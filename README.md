# RGBD-GAN

## requirements
```
chainer >= 7.0.0
cupy >= 7.0.0
```
and other

## dataset preparation
Resize all the training images to 128x128 and put them in an arbitrary directory.
For example,

```angular2html
image_dir
├- image0.png
├- image1.png
...

```

Then, set `image_path` in the config file to the path of training images.
Since a cache will be created during training, specify the directory where the cache will be created to `dataset_path` in the config file.

```
dataset_path: cache_dir_name
image_path: image_dir/*.png
```


## commands
Specify `dataset_path`, `image_path`, and `out` in config files.

```
python train_rgbd.py -g 0 --config configs/dcgan_shapenet_car.yml
python train_rgbd.py -g 0 --config configs/stylegan_shapenet_car.yml
python train_rgbd.py -g 0 --config configs/deepvoxels_shapenet_car.yml
```

## paper

https://arxiv.org/abs/1909.12573
