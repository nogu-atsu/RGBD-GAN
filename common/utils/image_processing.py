import chainer
import numpy as np
from chainer import cuda

try:
    import cupy
except:
    pass


def copy_to_cpu(imgs):
    if type(imgs) == chainer.variable.Variable:
        imgs = imgs.data
    try:
        if type(imgs) == cupy.core.core.ndarray:
            imgs = cuda.to_cpu(imgs)
    except:
        pass
    return imgs


def postprocessing_tanh(imgs):
    imgs = (imgs + 1) * 127.5
    imgs = np.clip(imgs, 0, 255)
    imgs = imgs.astype(np.uint8)
    return imgs


def postprocessing_sigmoid(imgs):
    imgs = imgs * 255.0
    imgs = np.clip(imgs, 0, 255)
    imgs = imgs.astype(np.uint8)
    return imgs
