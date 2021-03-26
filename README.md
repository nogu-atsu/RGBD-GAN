# RGBD-GAN

## requirements
```
chainer >= 7.0.0
cupy >= 7.0.0
```
and other

## commands
Specify `dataset_path`, `image_path`, and `out` in config files.

```
python train_rgbd.py -g 5 --config configs/dcgan_shapenet_car.yml
python train_rgbd.py -g 5 --config configs/stylegan_shapenet_car.yml
python train_rgbd.py -g 5 --config configs/deepvoxels_shapenet_car.yml
```

## paper

https://arxiv.org/abs/1909.12573
