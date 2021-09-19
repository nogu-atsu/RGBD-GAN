# RGBD-GAN

## requirements
```
numpy
Pillow
pyyaml
matplotlib
tqdm
chainer >= 7.0.0
cupy >= 7.0.0
```

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


## Training
Specify `dataset_path`, `image_path` in config files.
Specify `out` as the destination for saving models and generated images

```
python train_rgbd.py -g 0 --config configs/ffhq_stylegan_occlusion.yml
```

Generated images will be saved to `[out]/preview`

Depending on the initial value of the weights or seeds, the learning of 3D consistency may fail.

## paper

https://arxiv.org/abs/1909.12573
