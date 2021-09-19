import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


def feature_vector_normalization(x, eps=1e-8):
    # x: (B, C, H, W)
    alpha = 1.0 / F.sqrt(F.mean(x * x, axis=1, keepdims=True) + eps)
    return F.broadcast_to(alpha, x.data.shape) * x


class EqualizedConv2d(chainer.Chain):

    def __init__(self, in_ch, out_ch, ksize=1, stride=1, pad=0, nobias=False, gain=np.sqrt(2), lrmul=1):
        w = chainer.initializers.Normal(1.0 / lrmul)  # equalized learning rate
        self.inv_c = gain * np.sqrt(1.0 / (in_ch * ksize ** 2))
        self.inv_c = self.inv_c * lrmul
        super(EqualizedConv2d, self).__init__()
        with self.init_scope():
            self.c = L.Convolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w, nobias=nobias)

    def forward(self, x):
        return self.c(self.inv_c * x)


class EqualizedConv3d(chainer.Chain):

    def __init__(self, in_ch, out_ch, ksize=1, stride=1, pad=0, nobias=False, gain=np.sqrt(2), lrmul=1):
        w = chainer.initializers.Normal(1.0 / lrmul)  # equalized learning rate
        self.inv_c = gain * np.sqrt(1.0 / (in_ch * ksize ** 2))
        self.inv_c = self.inv_c * lrmul
        super(EqualizedConv3d, self).__init__()
        with self.init_scope():
            self.c = L.Convolution3D(in_ch, out_ch, ksize, stride, pad, initialW=w, nobias=nobias)

    def forward(self, x):
        return self.c(self.inv_c * x)

class EqualizedLinear(chainer.Chain):
    def __init__(self, in_ch, out_ch, initial_bias=None, nobias=False, gain=np.sqrt(2), lrmul=1):
        w = chainer.initializers.Normal(1.0 / lrmul)  # equalized learning rate
        self.inv_c = gain * np.sqrt(1.0 / in_ch)
        self.inv_c = self.inv_c * lrmul
        super(EqualizedLinear, self).__init__()
        with self.init_scope():
            self.c = L.Linear(in_ch, out_ch, initialW=w, initial_bias=initial_bias, nobias=nobias)

    def forward(self, x):
        return self.c(self.inv_c * x)

