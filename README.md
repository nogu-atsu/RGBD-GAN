# RGBD-GAN: Unsupervised 3D Representation Learning From Natural Image Datasets via RGBD Image Synthesis

[[ICLR2020]](https://openreview.net/forum?id=HyxjNyrtPr)

Author's official repository for RGBD-GAN.

## Results
RGBD image generation models conditioned on camera paremeters are trained on unlabeled RGB image datasets.

<img src="https://github.com/nogu-atsu/RGBD-GAN/blob/master/figs/overview.png">
(Odd rows: RGB, even rows: depth with colormap)


<img src="https://github.com/nogu-atsu/RGBD-GAN/blob/master/figs/output.gif">

## Pipeline
<img src="https://github.com/nogu-atsu/RGBD-GAN/blob/master/figs/pipeline.png" width="512">

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


## Citation
```
@inproceedings{RGBDGAN,
  title={RGBD-GAN: Unsupervised 3D Representation Learning From Natural Image Datasets via RGBD Image Synthesis},
  author={Noguchi, Atsuhiro and Harada, Tatsuya},
  booktitle={International Conference on Learning Representations},
  year={2020},
}
```