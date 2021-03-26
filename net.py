import sys
import os
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.link_hooks.spectral_normalization import SpectralNormalization
from chainer import Variable
import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

from common.networks.component.pggan import EqualizedConv2d, EqualizedLinear, feature_vector_normalization
from common.networks.component.auxiliary_links import LinkLeakyRelu
from common.networks.component.normalization.adain import AdaIN
from common.networks.component.scale import Scale
from common.networks.component.rescale import upscale2x, downscale2x, blur


class MappingNetwork(chainer.Chain):
    def __init__(self, ch=512):
        super().__init__()
        self.ch = ch
        with self.init_scope():
            self.l = chainer.ChainList(
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
            )
            self.ln = len(self.l)

    def make_hidden(self, batch_size):
        xp = self.xp
        if xp != np:
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1), dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1)).astype('f')
        if isinstance(chainer.global_config.dtype, chainer._Mixed16):
            z = z.astype("float16")
        return z

    def forward(self, x):
        h = feature_vector_normalization(x)
        for i in range(self.ln):
            h = self.l[i](h)
        return h


class NoiseBlock(chainer.Chain):
    def __init__(self, ch):
        super().__init__()
        with self.init_scope():
            self.b = Scale(axis=1, W_shape=ch, initialW=0)
        self.ch = ch

    def get_noise(self, batch_size, ch, shape):
        xp = self.xp
        if xp != np:
            z = xp.random.normal(size=(batch_size,) + shape, dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size,) + shape).astype('f')
        z = xp.broadcast_to(z, (ch, batch_size,) + shape)
        z = z.transpose((1, 0, 2, 3))
        return z

    def forward(self, h):
        batch_size = h.shape[0]
        noise = self.get_noise(batch_size, self.ch, h.shape[2:])
        h = h + self.b(noise)
        return h


class StyleBlock(chainer.Chain):
    def __init__(self, w_in, ch):
        super().__init__()
        self.w_in = w_in
        self.ch = ch
        with self.init_scope():
            self.s = EqualizedLinear(w_in, ch, initial_bias=chainer.initializers.One(), gain=1)
            self.b = EqualizedLinear(w_in, ch, initial_bias=chainer.initializers.Zero(), gain=1)

    def forward(self, w, h):
        ws = self.s(w)
        wb = self.b(w)
        return AdaIN(h, ws, wb)


class SynthesisBlock(chainer.Chain):
    def __init__(self, ch=512, ch_in=512, w_ch=512, upsample=True, enable_blur=False):
        super().__init__()
        self.upsample = upsample
        self.ch = ch
        self.ch_in = ch_in
        with self.init_scope():
            if not upsample:
                self.W = chainer.Parameter(shape=(ch_in, 4, 4))
                self.W.data[:] = 1  # w_data_tmp

            self.b0 = L.Bias(axis=1, shape=(ch,))
            self.b1 = L.Bias(axis=1, shape=(ch,))
            self.n0 = NoiseBlock(ch)
            self.n1 = NoiseBlock(ch)

            self.s0 = StyleBlock(w_ch, ch)
            self.s1 = StyleBlock(w_ch, ch)

            self.c0 = EqualizedConv2d(ch_in, ch, 3, 1, 1, nobias=True)
            self.c1 = EqualizedConv2d(ch, ch, 3, 1, 1, nobias=True)

        self.blur_k = None
        self.enable_blur = enable_blur

    def forward(self, w, x=None, add_noise=False):
        h = x
        batch_size, _ = w.shape
        if self.upsample:
            assert h is not None
            if self.blur_k is None:
                k = np.asarray([1, 2, 1]).astype('f')
                k = k[:, None] * k[None, :]
                k = k / np.sum(k)
                self.blur_k = self.xp.asarray(k)[None, None, :]
            if self.enable_blur:
                h = blur(upscale2x(h), self.blur_k)
            else:
                h = upscale2x(h)
            h = self.c0(h)
        else:
            h = F.broadcast_to(self.W, (batch_size, self.ch_in, 4, 4))

        # h should be (batch, ch, size, size)
        if add_noise:
            h = self.n0(h)

        h = F.leaky_relu(self.b0(h))
        h = self.s0(w, h)

        h = self.c1(h)
        if add_noise:
            h = self.n1(h)

        h = F.leaky_relu(self.b1(h))
        h = self.s1(w, h)
        return h


class StyleGenerator(chainer.Chain):
    def __init__(self, ch=512, enable_blur=False, rgbd=False, rotate_conv_input=False, use_encoder=False,
                 use_occupancy_net=False, initial_depth=1.0):
        super(StyleGenerator, self).__init__()
        self.max_stage = 17
        self.rgbd = rgbd
        self.rotate_conv_input = rotate_conv_input
        self.use_occupancy_net = use_occupancy_net
        out_ch = 4 if rgbd else 3
        with self.init_scope():
            self.blocks = chainer.ChainList(
                SynthesisBlock(ch, ch, ch, upsample=False),  # 4
                SynthesisBlock(ch, ch, ch, upsample=True, enable_blur=enable_blur),  # 8
                SynthesisBlock(ch, ch, ch, upsample=True, enable_blur=enable_blur),  # 16
                SynthesisBlock(ch, ch, ch, upsample=True, enable_blur=enable_blur),  # 32
                SynthesisBlock(ch // 2, ch, ch, upsample=True, enable_blur=enable_blur),  # 64
                SynthesisBlock(ch // 4, ch // 2, ch, upsample=True, enable_blur=enable_blur),  # 128
                # SynthesisBlock(ch // 8, ch // 4, ch, upsample=True, enable_blur=enable_blur),  # 256
                # SynthesisBlock(ch // 16, ch // 8, ch, upsample=True, enable_blur=enable_blur),  # 512
                # SynthesisBlock(ch // 32, ch // 16, ch, upsample=True, enable_blur=enable_blur)  # 1024
            )
            self.outs = chainer.ChainList(
                EqualizedConv2d(ch, out_ch, 1, 1, 0, gain=1),
                EqualizedConv2d(ch, out_ch, 1, 1, 0, gain=1),
                EqualizedConv2d(ch, out_ch, 1, 1, 0, gain=1),
                EqualizedConv2d(ch, out_ch, 1, 1, 0, gain=1),
                EqualizedConv2d(ch // 2, out_ch, 1, 1, 0, gain=1),
                EqualizedConv2d(ch // 4, out_ch, 1, 1, 0, gain=1),
                # EqualizedConv2d(ch // 8, out_ch, 1, 1, 0, gain=1),
                # EqualizedConv2d(ch // 16, out_ch, 1, 1, 0, gain=1),
                # EqualizedConv2d(ch // 32, out_ch, 1, 1, 0, gain=1)
            )
            if self.rgbd:
                if self.rotate_conv_input:
                    self.l1 = EqualizedLinear(9, ch)
                else:
                    self.l1 = EqualizedLinear(ch + 9, ch)
                self.l2 = EqualizedLinear(ch, ch)
            if rotate_conv_input:
                self.rotate = StyleBlock(ch, ch)

            if use_encoder:
                self.enc = Encoder(ch, ch * 2, enable_blur=enable_blur)

            if use_occupancy_net:
                self.occupancy = OccupancyNet(in_ch=ch * 2 + 3, hidden_ch=32)

        # initialize depth weight to 0
        for out in self.outs:
            out.c.W.array[-1] = 0
            out.c.b.array[-1] = math.log(math.e ** initial_depth - 1)

        self.n_blocks = len(self.blocks)
        self.image_size = 128
        self.enable_blur = enable_blur

    def rotate_w(self, w, theta):
        w = F.concat([w, theta * 16])
        w = F.leaky_relu(self.l1(w))
        w = F.leaky_relu(self.l2(w))
        return w

    def w_from_theta(self, theta):
        w_ = theta
        w_ = F.leaky_relu(self.l1(w_))
        w_ = F.leaky_relu(self.l2(w_))
        return w_

    def forward(self, w, w2, stage, theta=None, add_noise=True, return_feature=False):
        '''
            for alpha in [0, 1), and 2*k+2 + alpha < self.max_stage (-1 <= k <= ...):
            stage 0 + alpha       : z ->        block[0] -> out[0] * 1
            stage 2*k+1 + alpha   : z -> ... -> block[k] -> (up -> out[k]) * (1 - alpha)
                                    .................... -> (block[k+1] -> out[k+1]) * (alpha)
            stage 2*k+2 + alpha   : z -> ............... -> (block[k+1] -> out[k+1]) * 1
            over flow stages continues.
        '''
        # theta: (batchsize, )
        # if self.rgbd:
        add_noise = False

        stage = min(stage, self.max_stage - 1e-8)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        if self.rgbd and theta is None:
            assert False, "theta is None"

        h = None
        if stage % 2 == 0:
            k = (stage - 2) // 2

            for i in range(0, (k + 1) + 1):  # 0 .. k+1
                if i == 3:  # resolution 32~
                    w = w2
                if self.rgbd and i < 2:
                    if self.rotate_conv_input:
                        _w = self.w_from_theta(theta)
                    else:
                        _w = self.rotate_w(w, theta)
                else:
                    _w = w
                h = self.blocks[i](_w, x=h, add_noise=add_noise)
                if return_feature and i == 3:
                    feat = h

            h = self.outs[k + 1](h)

        else:
            k = (stage - 1) // 2

            for i in range(0, k + 1):  # 0 .. k
                if i == 3:
                    w = w2
                if self.rgbd and i < 2:
                    if self.rotate_conv_input:
                        _w = self.w_from_theta(theta)
                    else:
                        _w = self.rotate_w(w, theta)
                else:
                    _w = w
                h = self.blocks[i](_w, x=h, add_noise=add_noise)
                if return_feature and i == 3:
                    feat = h

            h_0 = upscale2x(self.outs[k](h))
            h_1 = self.outs[k + 1](self.blocks[k + 1](w, x=h, add_noise=add_noise))
            assert 0. <= alpha < 1.
            h = (1.0 - alpha) * h_0 + alpha * h_1

        if self.rgbd:
            # inverse depth
            depth = 1 / (F.softplus(h[:, -1:]) + 1e-4)
            # print(depth.array.mean(), depth.array.std())
            h = h[:, :3]
            h = F.concat([h, depth])
        if chainer.configuration.config.train:
            if return_feature:
                return h, feat
            else:
                return h
        else:
            min_sample_image_size = 64
            if h.data.shape[2] < min_sample_image_size:  # too small
                scale = int(min_sample_image_size // h.data.shape[2])
                return F.unpooling_2d(h, scale, scale, 0, outsize=(min_sample_image_size, min_sample_image_size))
            else:
                return h


class StyleGANGenerator(chainer.Chain):
    def __init__(self, ch, enable_blur=False, rgbd=False, rotate_conv_input=False, use_encoder=False,
                 use_occupancy_net=False, initial_depth=None):
        super(StyleGANGenerator, self).__init__()
        self.ch = ch
        rotate_conv_input = True if rotate_conv_input else False
        if initial_depth is None:
            initial_depth = 1.0
        with self.init_scope():
            self.mapping = MappingNetwork(ch)
            self.gen = StyleGenerator(ch, enable_blur=enable_blur, rgbd=rgbd, rotate_conv_input=rotate_conv_input,
                                      use_encoder=use_encoder, use_occupancy_net=use_occupancy_net,
                                      initial_depth=initial_depth)
        if use_encoder:
            self.enc = self.gen.enc

        if use_occupancy_net:
            self.occupancy = self.gen.occupancy

    def make_hidden(self, batch_size):
        xp = self.xp
        if xp != np:
            z = xp.random.normal(size=(batch_size, self.ch * 2, 1, 1), dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size, self.ch * 2, 1, 1)).astype('f')
        z /= xp.sqrt(xp.sum(z * z, axis=1, keepdims=True) / self.ch + 1e-8)
        if isinstance(chainer.global_config.dtype, chainer._Mixed16):
            z = z.astype("float16")
        return z

    def forward(self, z, stage, theta=None, return_feature=False):
        if not isinstance(z, Variable):
            z = Variable(z)
        z, z2 = F.split_axis(z, 2, 1)  # for low_resolution and high resolution
        w = self.mapping(z)
        w2 = self.mapping(z2)
        x_fake = self.gen(w, w2=w2, stage=stage, theta=theta, return_feature=return_feature)
        if return_feature:
            return x_fake[0], x_fake[1]
        return x_fake


class DiscriminatorBlockBase(chainer.Chain):

    def __init__(self, ch, out_dim=1, sn=False):
        super(DiscriminatorBlockBase, self).__init__()
        with self.init_scope():
            if not sn:
                self.c0 = EqualizedConv2d(ch, ch, 3, 1, 1)
                self.c1 = EqualizedConv2d(ch, ch, 4, 1, 0)
                self.l2 = EqualizedLinear(ch, out_dim, gain=1)
            else:
                w = chainer.initializers.Uniform(1)
                self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w).add_hook(SpectralNormalization())
                self.c1 = L.Convolution2D(ch, ch, 4, 1, 0, initialW=w).add_hook(SpectralNormalization())
                self.l2 = L.Linear(ch, out_dim, initialW=w).add_hook(SpectralNormalization())

    def forward(self, x):
        h = x
        h = F.leaky_relu((self.c0(h)))
        h = F.leaky_relu((self.c1(h)))
        h = self.l2(h)
        return h


class DiscriminatorBlock(chainer.Chain):

    def __init__(self, in_ch, out_ch, enable_blur=False, sn=False, res=False, bn=False):
        super(DiscriminatorBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.res = res
        with self.init_scope():
            if not sn:
                self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
                self.c1 = EqualizedConv2d(out_ch, out_ch, 3, 1, 1)
                if res:
                    self.c_sc = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
            else:
                w = chainer.initializers.Uniform(1)
                self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w).add_hook(SpectralNormalization())
                self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w).add_hook(SpectralNormalization())
                if res:
                    self.c_sc = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w).add_hook(SpectralNormalization())
            if bn:
                self.b0 = L.BatchNormalization(out_ch)
                self.b1 = L.BatchNormalization(out_ch)
            else:
                self.b0 = lambda x: x
                self.b1 = lambda x: x
        self.blur_k = None
        self.enable_blur = enable_blur

    def forward(self, x):

        h = F.leaky_relu((self.b0(self.c0(x))))
        if self.res:
            shortcut = self.c_sc(x)
            h = self.b1(self.c1(h)) + shortcut
        else:
            h = self.b1(self.c1(h))
        h = F.leaky_relu(h)
        if self.blur_k is None:
            k = np.asarray([1, 2, 1]).astype('f')
            k = k[:, None] * k[None, :]
            k = k / np.sum(k)
            self.blur_k = self.xp.asarray(k)[None, None, :]
        if self.enable_blur:
            h = blur(downscale2x(h), self.blur_k)
        else:
            h = downscale2x(h)
        return h


class Discriminator(chainer.Chain):

    def __init__(self, ch=512, out_dim=1, enable_blur=False, sn=False, res=False):
        super(Discriminator, self).__init__()
        self.max_stage = 17
        self.sn = sn

        with self.init_scope():
            # NOTE: called in reversed order.
            self.blocks = chainer.ChainList(
                DiscriminatorBlockBase(ch, out_dim, sn=sn),  # 4
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn, res=res),  # 8
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn, res=res),  # 16
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn, res=res),  # 32
                DiscriminatorBlock(ch // 2, ch, enable_blur=enable_blur, sn=sn, res=res),  # 64
                DiscriminatorBlock(ch // 4, ch // 2, enable_blur=enable_blur, sn=sn, res=res),  # 128
            )
            # DiscriminatorBlock(ch // 8, ch // 4, enable_blur=enable_blur, sn=sn, res=res),  # 256
            # DiscriminatorBlock(ch // 16, ch // 8, enable_blur=enable_blur, sn=sn, res=res),  # 512
            # DiscriminatorBlock(ch // 32, ch // 16, enable_blur=enable_blur, sn=sn, res=res), )  # 1024

            if not sn:
                self.ins = chainer.ChainList(
                    EqualizedConv2d(3, ch, 1, 1, 0),
                    EqualizedConv2d(3, ch, 1, 1, 0),
                    EqualizedConv2d(3, ch, 1, 1, 0),
                    EqualizedConv2d(3, ch, 1, 1, 0),
                    EqualizedConv2d(3, ch // 2, 1, 1, 0),
                    EqualizedConv2d(3, ch // 4, 1, 1, 0),
                    # EqualizedConv2d(3, ch // 8, 1, 1, 0),
                    # EqualizedConv2d(3, ch // 16, 1, 1, 0),
                    # EqualizedConv2d(3, ch // 32, 1, 1, 0),
                )
            else:
                w = chainer.initializers.Uniform(1)
                self.ins = chainer.ChainList(
                    L.Convolution2D(3, ch, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch // 2, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch // 4, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    # L.Convolution2D(3, ch // 8, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    # L.Convolution2D(3, ch // 16, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    # L.Convolution2D(3, ch // 32, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                )

            self.enable_blur = enable_blur

    def forward(self, x, stage, return_hidden=False):
        '''
            for alpha in [0, 1), and 2*k+2 + alpha < self.max_stage (-1 <= k <= ...):
            stage 0 + alpha       : p <-        block[0] <- in[0] * 1
            stage 2*k+1 + alpha   : p <- ... <- block[k] <- (up <- in[k]) * (1 - alpha)
                                    .................... <- (block[k+1] <- in[k+1]) * (alpha)
            stage 2*k+2 + alpha   : p <- ............... <- (block[k+1] <- in[k+1]) * 1
            over flow stages continues.
        '''
        stage = min(stage, self.max_stage - 1e-8)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        h = x
        if stage % 2 == 0:
            k = (stage - 2) // 2
            h = F.leaky_relu(self.ins[k + 1](h))
            for i in reversed(range(0, (k + 1) + 1)):  # k+1 .. 0
                if i == 3:
                    feat = h  # for adversarial 3D consistency loss
                h = self.blocks[i](h)
        else:
            k = (stage - 1) // 2

            h_0 = F.leaky_relu(self.ins[k](downscale2x(h)))
            h_1 = self.blocks[k + 1](F.leaky_relu(self.ins[k + 1](x)))
            assert 0. <= alpha < 1.
            h = (1.0 - alpha) * h_0 + alpha * h_1

            for i in reversed(range(0, k + 1)):  # k .. 0
                if i == 3:
                    feat = h  # for adversarial 3D consistency loss
                h = self.blocks[i](h)
        if return_hidden:
            return h, feat
        return h


class DisentangledDiscriminator(chainer.Chain):

    def __init__(self, ch=512, enable_blur=False, sn=False, res=False, num_z=2):
        super(DisentangledDiscriminator, self).__init__()
        self.max_stage = 17
        self.sn = sn

        with self.init_scope():
            # NOTE: called in reversed order.
            self.shared_blocks = chainer.ChainList(
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn, res=res),  # 16
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn, res=res),  # 32
                DiscriminatorBlock(ch // 2, ch, enable_blur=enable_blur, sn=sn, res=res),  # 64
                DiscriminatorBlock(ch // 4, ch // 2, enable_blur=enable_blur, sn=sn, res=res),  # 128
                DiscriminatorBlock(ch // 8, ch // 4, enable_blur=enable_blur, sn=sn, res=res),  # 256
                DiscriminatorBlock(ch // 16, ch // 8, enable_blur=enable_blur, sn=sn, res=res),  # 512
                DiscriminatorBlock(ch // 32, ch // 16, enable_blur=enable_blur, sn=sn, res=res), )  # 1024

            self.camera_parameter_blocks = chainer.Sequential(  # camera parameter variant feature
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn, res=res),  # 8
                DiscriminatorBlockBase(ch, sn=sn, out_dim=9),  # 4  # Euler angles, translation vector
            )

            self.z_regression_blocks = chainer.Sequential(  # camera parameter invariant feature
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn, res=res),  # 8
                DiscriminatorBlockBase(ch, sn=sn, out_dim=ch * num_z),  # 4
            )

            self.discriminator_blocks = chainer.Sequential(  # camera parameter variant feature
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn, res=res),  # 8
                DiscriminatorBlockBase(ch, sn=sn),  # 4  # Euler angles*2, translation vector
            )

            if not sn:
                self.ins = chainer.ChainList(
                    EqualizedConv2d(3, ch, 1, 1, 0),
                    EqualizedConv2d(3, ch, 1, 1, 0),
                    EqualizedConv2d(3, ch, 1, 1, 0),
                    EqualizedConv2d(3, ch, 1, 1, 0),
                    EqualizedConv2d(3, ch // 2, 1, 1, 0),
                    EqualizedConv2d(3, ch // 4, 1, 1, 0),
                    EqualizedConv2d(3, ch // 8, 1, 1, 0),
                    EqualizedConv2d(3, ch // 16, 1, 1, 0),
                    EqualizedConv2d(3, ch // 32, 1, 1, 0), )
            else:
                w = chainer.initializers.Uniform(1)
                self.ins = chainer.ChainList(
                    L.Convolution2D(3, ch, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch // 2, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch // 4, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch // 8, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch // 16, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                    L.Convolution2D(3, ch // 32, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()), )
            self.camera_param_discriminator = CameraParamDiscriminator()
            self.enable_blur = enable_blur

    def forward(self, x, stage):
        '''
            for alpha in [0, 1), and 2*k+2 + alpha < self.max_stage (-1 <= k <= ...):
            stage 0 + alpha       : p <-        block[0] <- in[0] * 1
            stage 2*k+1 + alpha   : p <- ... <- block[k] <- (up <- in[k]) * (1 - alpha)
                                    .................... <- (block[k+1] <- in[k+1]) * (alpha)
            stage 2*k+2 + alpha   : p <- ............... <- (block[k+1] <- in[k+1]) * 1
            over flow stages continues.
        '''
        stage = min(stage, self.max_stage - 1e-8)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        h = x
        if stage % 2 == 0:
            k = (stage - 2) // 2
            h = F.leaky_relu(self.ins[k + 1](h))
            for i in reversed(range(0, k)):  # k+1 .. 0
                h = self.shared_blocks[i](h)
        else:
            k = (stage - 1) // 2

            h_0 = F.leaky_relu(self.ins[k](downscale2x(h)))
            h_1 = self.shared_blocks[k - 1](F.leaky_relu(self.ins[k + 1](x)))
            assert 0. <= alpha < 1.
            h = (1.0 - alpha) * h_0 + alpha * h_1

            for i in reversed(range(0, k - 1)):  # k .. 0
                h = self.shared_blocks[i](h)

        estimated_camera_parameter = self.camera_parameter_blocks(h)
        estimated_z = self.z_regression_blocks(h)
        h = self.discriminator_blocks(h)
        return h, estimated_camera_parameter, estimated_z


# pggan
class DCGANBlock(chainer.Chain):
    def __init__(self, ch=512, ch_in=512, upsample=True, enable_blur=False):
        super().__init__()
        self.upsample = upsample
        self.ch = ch
        self.ch_in = ch_in
        with self.init_scope():
            self.b0 = L.Bias(axis=1, shape=(ch,))
            self.b1 = L.Bias(axis=1, shape=(ch,))
            self.n0 = NoiseBlock(ch)
            self.n1 = NoiseBlock(ch)

            self.c0 = EqualizedConv2d(ch_in, ch, 3, 1, 1, nobias=True)
            self.c1 = EqualizedConv2d(ch, ch, 3, 1, 1, nobias=True)

        self.blur_k = None
        self.enable_blur = enable_blur

    def forward(self, x, add_noise=False):
        h = x

        if self.blur_k is None:
            k = np.asarray([1, 2, 1]).astype('f')
            k = k[:, None] * k[None, :]
            k = k / np.sum(k)
            self.blur_k = self.xp.asarray(k)[None, None, :]
        if self.enable_blur:
            h = blur(upscale2x(h), self.blur_k)
        else:
            h = upscale2x(h)
        h = self.c0(h)

        # h should be (batch, ch, size, size)
        if add_noise:
            h = self.n0(h)

        h = F.leaky_relu(self.b0(h))
        h = F.normalize(h)

        h = self.c1(h)
        if add_noise:
            h = self.n1(h)

        h = F.leaky_relu(self.b1(h))
        h = F.normalize(h)
        return h


class DCGANGenerator(chainer.Chain):
    def __init__(self, in_ch=128, ch=512, enable_blur=False, rgbd=False, use_encoder=False, use_occupancy_net=False,
                 initial_depth=None):
        super(DCGANGenerator, self).__init__()
        self.in_ch = in_ch
        self.ch = ch
        self.max_stage = 17
        self.rgbd = rgbd
        self.use_occupancy_net = use_occupancy_net
        out_ch = 4 if rgbd else 3
        if initial_depth is None:
            initial_depth = 1.0

        with self.init_scope():
            if self.rgbd:
                self.linear = EqualizedLinear(in_ch + 9, ch * 4 * 4)
            else:
                self.linear = EqualizedLinear(in_ch, ch * 4 * 4)
            self.blocks = chainer.ChainList(
                DCGANBlock(ch, ch, enable_blur=enable_blur),  # 8
                DCGANBlock(ch, ch, enable_blur=enable_blur),  # 16
                DCGANBlock(ch, ch, enable_blur=enable_blur),  # 32
                DCGANBlock(ch // 2, ch, enable_blur=enable_blur),  # 64
                DCGANBlock(ch // 4, ch // 2, enable_blur=enable_blur),  # 128
                # DCGANBlock(ch // 8, ch // 4, enable_blur=enable_blur),  # 256
                # DCGANBlock(ch // 16, ch // 8, enable_blur=enable_blur),  # 512
                # DCGANBlock(ch // 32, ch // 16, enable_blur=enable_blur)  # 1024
            )
            self.outs = chainer.ChainList(
                EqualizedConv2d(ch, out_ch, 1, 1, 0, gain=1),
                EqualizedConv2d(ch, out_ch, 1, 1, 0, gain=1),
                EqualizedConv2d(ch, out_ch, 1, 1, 0, gain=1),
                EqualizedConv2d(ch // 2, out_ch, 1, 1, 0, gain=1),
                EqualizedConv2d(ch // 4, out_ch, 1, 1, 0, gain=1),
                # EqualizedConv2d(ch // 8, out_ch, 1, 1, 0, gain=1),
                # EqualizedConv2d(ch // 16, out_ch, 1, 1, 0, gain=1),
                # EqualizedConv2d(ch // 32, out_ch, 1, 1, 0, gain=1)
            )
            if use_encoder:
                self.enc = Encoder(ch, in_ch, enable_blur=enable_blur)
            if use_occupancy_net:
                self.occupancy = OccupancyNet(in_ch=in_ch + 3, hidden_ch=32)

        # initialize depth weight to 0
        for out in self.outs:
            out.c.W.array[-1] = 0
            out.c.b.array[-1] = math.log(math.e ** initial_depth - 1)

        self.n_blocks = len(self.blocks)
        self.image_size = 128
        self.enable_blur = enable_blur

    def make_hidden(self, batch_size):
        xp = self.xp
        if xp != np:
            z = xp.random.normal(size=(batch_size, self.in_ch), dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size, self.in_ch)).astype('f')
        z /= xp.sqrt(xp.sum(z * z, axis=1, keepdims=True) / self.in_ch + 1e-8)
        if isinstance(chainer.global_config.dtype, chainer._Mixed16):
            z = z.astype("float16")
        return z

    def forward(self, z, stage, theta=None, style_mixing_rate=None, add_noise=True, return_feature=False):
        '''
            for alpha in [0, 1), and 2*k+2 + alpha < self.max_stage (-1 <= k <= ...):
            stage 0 + alpha       : z ->        block[0] -> out[0] * 1
            stage 2*k+1 + alpha   : z -> ... -> block[k] -> (up -> out[k]) * (1 - alpha)
                                    .................... -> (block[k+1] -> out[k+1]) * (alpha)
            stage 2*k+2 + alpha   : z -> ............... -> (block[k+1] -> out[k+1]) * 1
            over flow stages continues.
        '''
        # theta: (batchsize, )
        # if self.rgbd:
        add_noise = False

        stage = min(stage, self.max_stage - 1e-8)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        if self.rgbd and theta is None:
            assert False, "theta is None"

        if self.rgbd:
            h = F.concat([z, theta * 10])
        else:
            h = z

        h = self.linear(h).reshape(z.shape[0], self.ch, 4, 4)
        if stage % 2 == 0:
            k = (stage - 2) // 2
            for i in range(0, (k + 1)):  # 0 .. k+1
                h = self.blocks[i](x=h, add_noise=add_noise)
                if return_feature and i == 2:
                    feat = h

            h = self.outs[k](h)

        else:
            k = (stage - 1) // 2
            for i in range(0, k):  # 0 .. k
                h = self.blocks[i](x=h, add_noise=add_noise)
                if return_feature and i == 2:
                    feat = h

            h_0 = upscale2x(self.outs[k - 1](h))
            h_1 = self.outs[k](self.blocks[k](x=h, add_noise=add_noise))
            assert 0. <= alpha < 1.
            h = (1.0 - alpha) * h_0 + alpha * h_1

        if self.rgbd:
            # inverse depth
            depth = 1 / (F.softplus(h[:, -1:]) + 1e-4)
            # print(depth.array.mean(), depth.array.std())
            h = h[:, :3]
            h = F.concat([h, depth])
        if chainer.configuration.config.train:
            if return_feature:
                return h, feat
            else:
                return h
        else:
            min_sample_image_size = 64
            if h.data.shape[2] < min_sample_image_size:  # too small
                scale = int(min_sample_image_size // h.data.shape[2])
                return F.unpooling_2d(h, scale, scale, 0, outsize=(min_sample_image_size, min_sample_image_size))
            else:
                return h


# encoder
class EncoderBlockBase(chainer.Chain):
    def __init__(self, ch, dim_z=256):
        super(EncoderBlockBase, self).__init__()
        with self.init_scope():
            self.c0 = EqualizedConv2d(ch, ch, 3, 1, 1)
            self.c1 = EqualizedConv2d(ch, ch, 4, 1, 0)
            self.l2 = EqualizedLinear(ch, dim_z, gain=1)
            self.bn0 = L.BatchNormalization(ch)
            self.bn1 = L.BatchNormalization(ch)

    def forward(self, x):
        h = x
        h = F.leaky_relu((self.bn0(self.c0(h))))
        h = F.leaky_relu((self.bn1(self.c1(h))))
        h = self.l2(h)
        return h


class CameraParamGenerator(chainer.Chain):
    def __init__(self):
        super(CameraParamGenerator, self).__init__()
        net = [EqualizedLinear(8, 64),
               F.leaky_relu,
               EqualizedLinear(64, 64),
               F.leaky_relu,
               EqualizedLinear(64, 9)]
        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, z):
        camera_param = self.net(z)
        # normalize to cos**2+sin**2=1
        inv_norm = F.rsqrt(F.square(camera_param[:, :3]) + F.square(camera_param[:, 3:6]) + 1e-8)
        camera_param = F.concat([
            camera_param[:, :3] * inv_norm, camera_param[:, 3:6] * inv_norm, camera_param[:, 6:]
        ])
        return camera_param


class CameraParamDiscriminator(chainer.Chain):
    def __init__(self):
        super(CameraParamDiscriminator, self).__init__()
        net = [EqualizedLinear(9, 64),
               F.leaky_relu,
               EqualizedLinear(64, 64),
               F.leaky_relu,
               EqualizedLinear(64, 1)]
        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, camera_param):
        return self.net(camera_param)


class Encoder(chainer.Chain):
    def __init__(self, ch=512, dim_z=256, enable_blur=False, res=True):
        super(Encoder, self).__init__()
        self.max_stage = 17

        with self.init_scope():
            # NOTE: called in reversed order.
            self.blocks = chainer.ChainList(
                EncoderBlockBase(ch, dim_z + 9),  # 4
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=False, res=res, bn=True),  # 8
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=False, res=res, bn=True),  # 16
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=False, res=res, bn=True),  # 32
                DiscriminatorBlock(ch // 2, ch, enable_blur=enable_blur, sn=False, res=res, bn=True),  # 64
                DiscriminatorBlock(ch // 4, ch // 2, enable_blur=enable_blur, sn=False, res=res, bn=True),  # 128
            )
            # DiscriminatorBlock(ch // 8, ch // 4, enable_blur=enable_blur, sn=sn, res=res),  # 256
            # DiscriminatorBlock(ch // 16, ch // 8, enable_blur=enable_blur, sn=sn, res=res),  # 512
            # DiscriminatorBlock(ch // 32, ch // 16, enable_blur=enable_blur, sn=sn, res=res), )  # 1024

            self.ins = chainer.ChainList(
                EqualizedConv2d(3, ch, 1, 1, 0),
                EqualizedConv2d(3, ch, 1, 1, 0),
                EqualizedConv2d(3, ch, 1, 1, 0),
                EqualizedConv2d(3, ch, 1, 1, 0),
                EqualizedConv2d(3, ch // 2, 1, 1, 0),
                EqualizedConv2d(3, ch // 4, 1, 1, 0),
                # EqualizedConv2d(3, ch // 8, 1, 1, 0),
                # EqualizedConv2d(3, ch // 16, 1, 1, 0),
                # EqualizedConv2d(3, ch // 32, 1, 1, 0),
            )

            self.enable_blur = enable_blur

    def forward(self, x, stage):
        '''
            for alpha in [0, 1), and 2*k+2 + alpha < self.max_stage (-1 <= k <= ...):
            stage 0 + alpha       : p <-        block[0] <- in[0] * 1
            stage 2*k+1 + alpha   : p <- ... <- block[k] <- (up <- in[k]) * (1 - alpha)
                                    .................... <- (block[k+1] <- in[k+1]) * (alpha)
            stage 2*k+2 + alpha   : p <- ............... <- (block[k+1] <- in[k+1]) * 1
            over flow stages continues.
        '''
        stage = min(stage, self.max_stage - 1e-8)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        h = x
        if stage % 2 == 0:
            k = (stage - 2) // 2
            h = F.leaky_relu(self.ins[k + 1](h))
            for i in reversed(range(0, (k + 1) + 1)):  # k+1 .. 0
                h = self.blocks[i](h)
        else:
            k = (stage - 1) // 2

            h_0 = F.leaky_relu(self.ins[k](downscale2x(h)))
            h_1 = self.blocks[k + 1](F.leaky_relu(self.ins[k + 1](x)))
            assert 0. <= alpha < 1.
            h = (1.0 - alpha) * h_0 + alpha * h_1

            for i in reversed(range(0, k + 1)):  # k .. 0
                h = self.blocks[i](h)

        inv_norm = F.rsqrt(F.square(h[:, -9:-6]) + F.square(h[:, -6:-3]) + 1e-8)
        camera_param = F.concat([h[:, -9:-6] * inv_norm, h[:, -6:-3] * inv_norm, h[:, -3:]])
        return h[:, :-9], camera_param


class MLP(chainer.Chain):
    def __init__(self, ch, out_ch, sn=False):
        super(MLP, self).__init__()
        w = chainer.initializers.Uniform(1)
        if sn:
            net = [
                L.Linear(ch, ch, initialW=w).add_hook(SpectralNormalization()),
                F.leaky_relu,
                L.Linear(ch, ch, initialW=w).add_hook(SpectralNormalization()),
                F.leaky_relu,
                L.Linear(ch, out_ch, initialW=w).add_hook(SpectralNormalization()),
            ]
        else:
            net = [
                EqualizedLinear(ch, ch),
                F.leaky_relu,
                EqualizedLinear(ch, ch),
                F.leaky_relu,
                EqualizedLinear(ch, out_ch),
            ]
        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class BigBiGANDiscriminator(chainer.Chain):
    def __init__(self, ch, dim_z, enable_blur=False, sn=False, res=False):
        super(BigBiGANDiscriminator, self).__init__()
        self.sn = sn
        w = chainer.initializers.Uniform(1)
        with self.init_scope():
            self.f = Discriminator(ch, ch, enable_blur=enable_blur, sn=sn, res=res)
            self.h_z = MLP(dim_z, ch, sn=sn)
            self.h_cp = MLP(9, ch, sn=sn)
            self.j = MLP(ch * 3, 1, sn=sn)
            if sn:
                self.s_x = L.Linear(ch, 1, initialW=w).add_hook(SpectralNormalization())
                self.s_z = L.Linear(ch, 1, initialW=w).add_hook(SpectralNormalization())
                self.s_cp = L.Linear(ch, 1, initialW=w).add_hook(SpectralNormalization())
            else:
                self.s_x = EqualizedLinear(ch, 1)
                self.s_z = EqualizedLinear(ch, 1)
                self.s_cp = EqualizedLinear(ch, 1)

    def forward(self, x, z, cp, stage):
        f = self.f(x, stage)
        h_z = self.h_z(z)
        h_cp = self.h_cp(cp)
        s_xzcp = self.j(F.concat([f, h_z, h_cp], axis=1))
        s_x = self.s_x(f)
        s_z = self.s_z(h_z)
        s_cp = self.s_cp(h_cp)
        return s_xzcp, s_x, s_z, s_cp
        # return s_xzcp * 1e-5, s_x, s_z * 1e-5, s_cp * 1e-5


class OccupancyNet(chainer.Chain):
    def __init__(self, in_ch, hidden_ch):
        super(OccupancyNet, self).__init__()
        net = [EqualizedLinear(in_ch, hidden_ch),
               F.leaky_relu,
               EqualizedLinear(hidden_ch, hidden_ch),
               F.leaky_relu,
               EqualizedLinear(hidden_ch, 1)
               ]
        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, z, coords):
        """
        calculate occupancy field for the real-world coordinate
        :param z: latent vector
        :param coords: real-world coordinate, ()
        :return:
        """
        h = F.concat([F.tile(z[:, :, None], (1, 1, coords.shape[2])), coords * z.shape[1] ** 0.5])
        h = h.transpose(0, 2, 1).reshape(-1, z.shape[1] + 3)
        occupancy_field = self.net(h)
        return occupancy_field
