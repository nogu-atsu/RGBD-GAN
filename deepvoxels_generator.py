import math
import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Variable
from chainer.link_hooks.spectral_normalization import SpectralNormalization

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

from common.networks.component.pggan import EqualizedConv2d, EqualizedLinear, feature_vector_normalization, \
    EqualizedConv3d
from common.networks.component.auxiliary_links import LinkLeakyRelu
from common.networks.component.normalization.adain import AdaIN
from common.networks.component.scale import Scale
from common.networks.component.rescale import upscale2x, downscale2x, blur, upscale2x3d, blur3d

from deepvoxel.deepvoxel import DeepVoxels
from deepvoxel.projection import ProjectionHelper

from net import CameraParamGenerator, SynthesisBlock


class MappingNetwork3D(chainer.Chain):
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
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1, 1), dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1, 1)).astype('f')
        if isinstance(chainer.global_config.dtype, chainer._Mixed16):
            z = z.astype("float16")
        return z

    def forward(self, x):
        h = feature_vector_normalization(x)
        for i in range(self.ln):
            h = self.l[i](h)
        return h


class NoiseBlock(chainer.Chain):
    # same as stylegan
    def __init__(self, ch):
        super().__init__()
        with self.init_scope():
            self.b = Scale(axis=1, W_shape=ch, initialW=0)
        self.ch = ch

    def get_noise(self, batch_size, ch, shape):
        xp = self.xp
        if xp != np:
            z = xp.random.normal(size=(batch_size, 1) + shape, dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size, 1) + shape).astype('f')
        z = xp.broadcast_to(z, (batch_size, ch) + shape)
        return z

    def forward(self, h):
        batch_size = h.shape[0]
        noise = self.get_noise(batch_size, self.ch, h.shape[2:])
        h = h + self.b(noise)
        return h


class StyleBlock(chainer.Chain):
    # same as stylegan
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


class SynthesisBlock3D(chainer.Chain):
    def __init__(self, ch=512, ch_in=512, w_ch=512, upsample=True, enable_blur=False):
        super().__init__()
        self.upsample = upsample
        self.ch = ch
        self.ch_in = ch_in
        with self.init_scope():
            if not upsample:
                self.W = chainer.Parameter(shape=(ch_in, 4, 4, 4))
                self.W.data[:] = 1  # w_data_tmp

            self.b0 = L.Bias(axis=1, shape=(ch,))
            self.b1 = L.Bias(axis=1, shape=(ch,))
            self.n0 = NoiseBlock(ch)
            self.n1 = NoiseBlock(ch)

            self.s0 = StyleBlock(w_ch, ch)
            self.s1 = StyleBlock(w_ch, ch)

            self.c0 = EqualizedConv3d(ch_in, ch, 3, 1, 1, nobias=True)
            self.c1 = EqualizedConv3d(ch, ch, 3, 1, 1, nobias=True)

        self.blur_k = None
        self.enable_blur = enable_blur

    def forward(self, w, x=None, add_noise=False):
        h = x
        batch_size, _ = w.shape
        if self.upsample:
            assert h is not None
            if self.blur_k is None:
                k = np.asarray([1, 2, 1]).astype('f')
                k = k[:, None, None] * k[None, :, None] * k[None, None, :]
                k = k / np.sum(k)
                self.blur_k = self.xp.asarray(k)[None, None, :]
            if self.enable_blur:
                h = blur3d(upscale2x3d(h), self.blur_k)
            else:
                h = upscale2x3d(h)
            h = self.c0(h)
        else:
            h = F.broadcast_to(self.W, (batch_size, self.ch_in, 4, 4, 4))

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


class VoxelGenerator(chainer.Chain):
    def __init__(self, ch, ch_out):
        super(VoxelGenerator, self).__init__()
        with self.init_scope():
            self.net = chainer.ChainList(
                SynthesisBlock3D(ch // 4, ch // 4, ch, upsample=False),  # 4
                SynthesisBlock3D(ch // 4, ch // 4, ch, upsample=True, enable_blur=False),  # 8
                SynthesisBlock3D(ch // 8, ch // 4, ch, upsample=True, enable_blur=False),  # 16
                SynthesisBlock3D(ch // 8, ch // 8, ch, upsample=True, enable_blur=False),  # 32
            )
            self.out = EqualizedConv3d(ch // 8, ch_out, 1, 1, 0)

    def forward(self, w):
        h = None
        for l in self.net:
            h = l(w, h)
        h = self.out(h)
        return h  # b x ch_out x 32 x 32 x 32


class StyleGenerator(chainer.Chain):
    def __init__(self, w_ch, in_ch, hidden_ch=256):
        super(StyleGenerator, self).__init__()
        with self.init_scope():
            self.c0 = EqualizedConv2d(in_ch, hidden_ch * 2, 4, 2, 1)
            self.c1 = EqualizedConv2d(hidden_ch * 2, hidden_ch * 4, 4, 2, 1)
            self.c4 = EqualizedConv2d(hidden_ch * 4, hidden_ch * 4, 3, 1, 1)
            self.c5 = EqualizedConv2d(hidden_ch * 4, hidden_ch * 2, 3, 1, 1)
            self.c6 = EqualizedConv2d(hidden_ch * 2 * 2, hidden_ch, 3, 1, 1)
            self.c7 = EqualizedConv2d(hidden_ch + in_ch, 3, 3, 1, 1, gain=0.5)

            self.s0 = StyleBlock(w_ch, hidden_ch * 2)
            self.s1 = StyleBlock(w_ch, hidden_ch * 4)
            self.s4 = StyleBlock(w_ch, hidden_ch * 4)
            self.s5 = StyleBlock(w_ch, hidden_ch * 2)
            self.s6 = StyleBlock(w_ch, hidden_ch)

    def __call__(self, h, w, stage):
        h1 = F.leaky_relu(self.c0(h))
        h1 = self.s0(w, h1)
        h2 = F.leaky_relu(self.c1(h1))
        h2 = self.s1(w, h2)
        h3 = F.leaky_relu(self.c4(h2))
        h3 = self.s4(w, h3)
        h3 = upscale2x(h3)
        h3 = F.leaky_relu(self.c5(h3))
        h3 = F.concat([self.s5(w, h3), h1])
        h3 = upscale2x(h3)
        h3 = F.leaky_relu(self.c6(h3))
        h3 = F.concat([self.s6(w, h3), h])
        h = self.c7(h3)
        return h


class Generator(chainer.Chain):
    def __init__(self, ch, occlusion_type="deepvoxels", background_generator=False, config=None):
        super(Generator, self).__init__()
        self.ch = ch
        self.use_background_generator = background_generator
        scale = 0.5  # 1 / focal lengthくらい？
        grid_dim = 32
        near_plane = np.sqrt(3) / 4
        lift_intrinsic = np.array([[64 * 2., 0.0, 32., 0.0],
                                   [0.0, 64 * 2., 32., 0.0],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]])
        voxel_size = (1. / grid_dim) * 1.1 * scale
        depth_max = grid_dim * voxel_size + near_plane
        grid_dims = 3 * [grid_dim]
        proj_image_dims = [64, 64]
        frustrum_depth = int(np.ceil(np.sqrt(3) * grid_dims[-1]))
        num_grid_feats = 32
        self.projection = ProjectionHelper(projection_intrinsic=lift_intrinsic,
                                           lifting_intrinsic=lift_intrinsic,
                                           depth_min=0.,
                                           depth_max=depth_max,
                                           projection_image_dims=proj_image_dims,
                                           lifting_image_dims=proj_image_dims,
                                           grid_dims=grid_dims,
                                           voxel_size=voxel_size,
                                           device=None,
                                           frustrum_depth=frustrum_depth,
                                           near_plane=near_plane)
        with self.init_scope():
            self.voxel_gen = VoxelGenerator(ch, num_grid_feats)
            self.deepvoxel = DeepVoxels(lifting_img_dims=proj_image_dims,
                                        frustrum_img_dims=proj_image_dims,
                                        grid_dims=grid_dims,
                                        occlusion_type=occlusion_type,
                                        num_grid_feats=num_grid_feats,
                                        nf0=64,
                                        img_sidelength=64,
                                        voxel_size=voxel_size,
                                        near_plane=near_plane,
                                        config=config)
            self.style_generator = StyleGenerator(w_ch=ch, in_ch=num_grid_feats)
            self.camera_param_generator = CameraParamGenerator()
            if self.use_background_generator:
                self.background_generator = BackgroundFeatureGenerator(ch, num_grid_feats)

        self.mapping = MappingNetwork3D(ch)

    def make_hidden(self, batch_size):
        xp = self.xp
        if xp != np:
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1), dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1)).astype('f')
        z /= xp.sqrt(xp.sum(z * z, axis=1, keepdims=True) / self.ch + 1e-8)
        if isinstance(chainer.global_config.dtype, chainer._Mixed16):
            z = z.astype("float16")
        return z

    def forward(self, z, stage, camera_matrices, z2=None, z3=None, z4=None, theta=None):
        # z1 and z2 are for foreground, z3 and z4 are for background
        proj_mappings = list()
        for i in range(len(camera_matrices)):
            proj_mappings.append(self.projection.compute_proj_idcs(camera_matrices[i]))

        proj_frustrum_idcs, proj_grid_coords = list(zip(*proj_mappings))
        if not isinstance(z, Variable):
            z = Variable(z)
        if not isinstance(z2, Variable):
            z2 = Variable(z2)

        w = self.mapping(z)
        voxel = self.voxel_gen(w)
        img_feature = self.deepvoxel(proj_frustrum_idcs, proj_grid_coords, voxel,
                                     return_foreground_weight=self.use_background_generator)
        if self.use_background_generator:
            novel_feats, depth, foreground_weight = img_feature
            if z3 is None:
                z3 = Variable(self.make_hidden(z.shape[0]))
                z4 = Variable(self.make_hidden(z.shape[0]))
            w3 = self.mapping(z3)
            w4 = self.mapping(z4)
            background, background_depth = self.background_generator(w3, w4, theta)
            novel_feats = F.normalize(novel_feats, axis=1) + \
                          F.normalize(background, axis=1) * (1 - foreground_weight)
            depth = depth + background_depth * (1 - foreground_weight)
            print(foreground_weight.array.mean(), foreground_weight.array.std(),
                  depth.array.mean(), background_depth.mean(),
                  novel_feats.array.std(), background.array.std())
        else:
            novel_feats, depth = img_feature

        if z2 is None:
            z2 = self.make_hidden(z.shape[0])
        w2 = self.mapping(z2)
        novel_img = self.style_generator(novel_feats, w2, stage)
        x_fake = F.concat([novel_img, depth], axis=1)
        return x_fake


class DiscriminatorBlockBase(chainer.Chain):

    def __init__(self, ch, sn=False):
        super(DiscriminatorBlockBase, self).__init__()
        with self.init_scope():
            if not sn:
                self.c0 = EqualizedConv2d(ch, ch, 3, 1, 1)
                self.c1 = EqualizedConv2d(ch, ch, 4, 1, 0)
                self.l2 = EqualizedLinear(ch, 1, gain=1)
            else:
                w = chainer.initializers.GlorotUniform(math.sqrt(2))
                self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w).add_hook(SpectralNormalization())
                self.c1 = L.Convolution2D(ch, ch, 4, 1, 0, initialW=w).add_hook(SpectralNormalization())
                self.l2 = L.Linear(ch, 1, initialW=w).add_hook(SpectralNormalization())

    def forward(self, x):
        h = x
        h = F.leaky_relu((self.c0(h)))
        h = F.leaky_relu((self.c1(h)))
        h = self.l2(h)
        return h


class DiscriminatorBlock(chainer.Chain):
    # same as net.DiscriminatorBlock(res=True)

    def __init__(self, in_ch, out_ch, enable_blur=False, sn=False):
        super(DiscriminatorBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        with self.init_scope():
            if not sn:
                self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
                self.c1 = EqualizedConv2d(out_ch, out_ch, 3, 1, 1)
                self.c_sc = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
            else:
                w = chainer.initializers.Uniform(1)
                self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w).add_hook(SpectralNormalization())
                self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w).add_hook(SpectralNormalization())
                self.c_sc = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w).add_hook(SpectralNormalization())

        self.blur_k = None
        self.enable_blur = enable_blur

    def forward(self, x):
        shortcut = self.c_sc(x)

        res = F.leaky_relu((self.c0(x)))
        h = F.leaky_relu((self.c1(res) + shortcut))
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

    def __init__(self, ch=512, enable_blur=False, sn=False):
        super(Discriminator, self).__init__()
        self.max_stage = 17
        self.sn = sn

        with self.init_scope():
            # NOTE: called in reversed order.
            self.blocks = chainer.ChainList(
                DiscriminatorBlockBase(ch, sn=sn),
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn),
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn),
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur, sn=sn),
                DiscriminatorBlock(ch // 2, ch, enable_blur=enable_blur, sn=sn),
            )
            if not sn:
                self.ins = chainer.ChainList(
                    EqualizedConv2d(3, ch // 2, 1, 1, 0),
                )
            else:
                w = chainer.initializers.GlorotUniform(math.sqrt(2))
                self.ins = chainer.ChainList(
                    L.Convolution2D(3, ch // 2, 1, 1, 0, initialW=w).add_hook(SpectralNormalization()),
                )

            self.enable_blur = enable_blur

    def forward(self, x):
        '''
            for alpha in [0, 1), and 2*k+2 + alpha < self.max_stage (-1 <= k <= ...):
            stage 0 + alpha       : p <-        block[0] <- in[0] * 1
            stage 2*k+1 + alpha   : p <- ... <- block[k] <- (up <- in[k]) * (1 - alpha)
                                    .................... <- (block[k+1] <- in[k+1]) * (alpha)
            stage 2*k+2 + alpha   : p <- ............... <- (block[k+1] <- in[k+1]) * 1
            over flow stages continues.
        '''
        h = x
        h = F.leaky_relu(self.ins[0](h))

        for i in reversed(range(0, 5)):
            h = self.blocks[i](h)

        return h


class BackgroundFeatureGenerator(chainer.Chain):
    # generate virtual infinity distance background
    def __init__(self, ch=512, out_ch=64, enable_blur=False):
        super(BackgroundFeatureGenerator, self).__init__()
        self.img_size = 64
        self.background_depth = 4  # virtual background distance

        x_pos, y_pos = np.meshgrid(np.arange(self.img_size) - self.img_size // 2,
                                   np.arange(self.img_size) - self.img_size // 2)
        depth_map = self.background_depth * \
                    self.img_size * 2 / np.sqrt((self.img_size * 2) ** 2 + x_pos ** 2 + y_pos ** 2)

        with self.init_scope():
            self.blocks = chainer.ChainList(
                SynthesisBlock(ch, ch, ch, upsample=False),  # 4
                SynthesisBlock(ch, ch, ch, upsample=True, enable_blur=enable_blur),  # 8
                SynthesisBlock(ch, ch, ch, upsample=True, enable_blur=enable_blur),  # 16
                SynthesisBlock(ch, ch, ch, upsample=True, enable_blur=enable_blur),  # 32
                SynthesisBlock(ch // 2, ch, ch, upsample=True, enable_blur=enable_blur),  # 64
            )
            self.conv = EqualizedConv2d(ch // 2, out_ch, 1, 1, 0, gain=1)
            self.l1 = EqualizedLinear(ch + 9, ch)
            self.l2 = EqualizedLinear(ch, ch)

        self.add_persistent("depth_map", depth_map)

        self.n_blocks = len(self.blocks)
        self.image_size = 64
        self.enable_blur = enable_blur

    def rotate_w(self, w, theta):
        w = F.concat([w, theta * 16])
        w = F.leaky_relu(self.l1(w))
        w = F.leaky_relu(self.l2(w))
        return w

    def forward(self, w, w2, theta=None):
        '''
            for alpha in [0, 1), and 2*k+2 + alpha < self.max_stage (-1 <= k <= ...):
            stage 0 + alpha       : z ->        block[0] -> out[0] * 1
            stage 2*k+1 + alpha   : z -> ... -> block[k] -> (up -> out[k]) * (1 - alpha)
                                    .................... -> (block[k+1] -> out[k+1]) * (alpha)
            stage 2*k+2 + alpha   : z -> ............... -> (block[k+1] -> out[k+1]) * 1
            over flow stages continues.
        '''
        # theta: (batchsize, )

        h = None

        for i in range(0, 5):  # 0 .. k+1
            if i == 3:  # resolution 32~
                w = w2
            if i < 2:
                _w = self.rotate_w(w, theta)
            else:
                _w = w
            h = self.blocks[i](_w, x=h, add_noise=False)
        h = self.conv(h)
        depth = self.xp.tile(self.depth_map[None, None], (w.shape[0], 1, 1, 1))
        return h, depth
