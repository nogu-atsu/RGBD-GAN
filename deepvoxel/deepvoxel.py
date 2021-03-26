import chainer
from chainer import links as L
from chainer import functions as F
from chainer.backends import cuda

import numpy as np

from deepvoxel.projection import ProjectionHelper
from common.networks.component.pggan import EqualizedConv3d, EqualizedConv2d, EqualizedLinear
from net import SynthesisBlock


class ReplicationPad:
    def __init__(self, pad_width):
        if isinstance(pad_width, tuple):
            self.pad_width = ((0, 0), (0, 0)) + pad_width
        else:
            self.pad_width = pad_width

    def __call__(self, x):
        if isinstance(self.pad_width, int):
            self.pad_width = ((0, 0), (0, 0)) + ((self.pad_width, self.pad_width),) * (x.ndim - 2)
        return F.pad(x, pad_width=self.pad_width, mode='edge')


class ReflectionPad:
    def __init__(self, pad_width):
        if isinstance(pad_width, tuple):
            self.pad_width = ((0, 0), (0, 0)) + pad_width
        else:
            self.pad_width = pad_width

    def __call__(self, x):
        if isinstance(self.pad_width, int):
            self.pad_width = ((0, 0), (0, 0)) + ((self.pad_width, self.pad_width),) * (x.ndim - 2)
        return F.pad(x, pad_width=self.pad_width, mode='reflect')


class Conv3dSame(chainer.Chain):
    '''3D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=None):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        if padding_layer is not None:
            assert False, "padding layer is not supported"
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        with self.init_scope():
            self.net = chainer.Sequential(
                ReflectionPad(pad_width=((ka, kb), (ka, kb), (ka, kb))),
                EqualizedConv3d(in_channels, out_channels, kernel_size, stride=1, nobias=not bias)
            )

    def forward(self, x):
        return self.net(x)


class UnetSkipConnectionBlock3d(chainer.Chain):
    '''Helper class for building a 3D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 norm=L.BatchNormalization,
                 submodule=None):
        super().__init__()

        if submodule is None:
            model = [DownBlock3D(outer_nc, inner_nc, norm=norm),
                     UpBlock3D(inner_nc, outer_nc, norm=norm)]
        else:
            model = [DownBlock3D(outer_nc, inner_nc, norm=norm),
                     submodule,
                     UpBlock3D(2 * inner_nc, outer_nc, norm=norm)]

        with self.init_scope():
            self.model = chainer.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return F.concat([x, forward_passed], axis=1)


class DownBlock3D(chainer.Chain):
    '''A 3D convolutional downsampling block.
    '''

    def __init__(self, in_channels, out_channels, norm=L.BatchNormalization):
        super().__init__()

        net = [
            ReflectionPad(pad_width=1),
            EqualizedConv3d(in_channels,
                            out_channels,
                            ksize=4,
                            stride=2,
                            pad=0,
                            nobias=True if norm is not None else False),
        ]

        if norm is not None:
            net += [norm(out_channels)]

        net += [F.leaky_relu]
        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class UpBlock3D(chainer.Chain):
    '''A 3D convolutional upsampling block.
    '''

    def __init__(self, in_channels, out_channels, norm=L.BatchNormalization):
        super().__init__()

        net = [
            L.Deconvolution3D(in_channels,
                              out_channels,
                              ksize=4,
                              stride=2,
                              pad=1,
                              nobias=True if norm is not None else False),
        ]

        if norm is not None:
            net += [norm(out_channels)]

        net += [F.relu]
        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = F.concat([skipped, x], axis=1)
        else:
            input = x
        return self.net(input)


class Conv2dSame(chainer.Chain):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=None):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        assert padding_layer is None, "this padding is not supported"
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        with self.init_scope():
            self.net = chainer.Sequential(
                ReflectionPad(pad_width=((ka, kb), (ka, kb))),
                EqualizedConv2d(in_channels, out_channels, kernel_size, nobias=not bias, stride=1)
            )

        # self.weight = self.net[1].W
        # self.bias = self.net[1].b

    def forward(self, x):
        return self.net(x)


class UpBlock(chainer.Chain):
    '''A 2d-conv upsampling block with a variety of options for upsampling, and following best practices / with
    reasonable defaults. (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 post_conv=True,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=L.BatchNormalization,
                 upsampling_mode='transpose'):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param post_conv: Whether to have another convolutional layer after the upsampling layer.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param upsampling_mode: Which upsampling mode:
                transpose: Upsampling with stride-2, kernel size 4 transpose convolutions.
                bilinear: Feature map is upsampled with bilinear upsampling, then a conv layer.
                nearest: Feature map is upsampled with nearest neighbor upsampling, then a conv layer.
                shuffle: Feature map is upsampled with pixel shuffling, then a conv layer.
        '''
        super().__init__()

        net = list()

        if upsampling_mode == 'transpose':
            net += [L.Deconvolution2D(in_channels,
                                      out_channels,
                                      ksize=4,
                                      stride=2,
                                      pad=1,
                                      nobias=False if norm is None else True)]
        # elif upsampling_mode == 'bilinear':
        #     net += [nn.UpsamplingBilinear2d(scale_factor=2)]
        #     net += [
        #         Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        # elif upsampling_mode == 'nearest':
        #     net += [nn.UpsamplingNearest2d(scale_factor=2)]
        #     net += [
        #         Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        # elif upsampling_mode == 'shuffle':
        #     net += [nn.PixelShuffle(upscale_factor=2)]
        #     net += [
        #         Conv2dSame(in_channels // 4, out_channels, kernel_size=3,
        #                    bias=True if norm is None else False)]
        else:
            raise ValueError("Unknown upsampling mode!")

        if norm is not None:
            net += [norm(out_channels)]

        net += [F.relu]

        if use_dropout:
            net += [lambda x: F.dropout(x, dropout_prob)]

        if post_conv:
            net += [Conv2dSame(out_channels,
                               out_channels,
                               kernel_size=3,
                               bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(out_channels)]

            net += [F.relu]

            if use_dropout:
                net += [lambda x: F.dropout(x, 0.1)]

        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = F.concat([skipped, x], axis=1)
        else:
            input = x
        return self.net(input)


class DownBlock(chainer.Chain):
    '''A 2D-conv downsampling block following best practices / with reasonable defaults
    (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 prep_conv=True,
                 middle_channels=None,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=L.BatchNormalization):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param prep_conv: Whether to have another convolutional layer before the downsampling layer.
        :param middle_channels: If prep_conv is true, this sets the number of channels between the prep and downsampling
                                convs.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        '''
        super().__init__()

        if middle_channels is None:
            middle_channels = in_channels

        net = list()

        if prep_conv:
            net += [ReflectionPad(pad_width=1),
                    EqualizedConv2d(in_channels,
                                    middle_channels,
                                    ksize=3,
                                    pad=0,
                                    stride=1,
                                    nobias=False if norm is None else True)]

            if norm is not None:
                net += [norm(middle_channels)]

            net += [F.leaky_relu]

            if use_dropout:
                net += [lambda x: F.dropout(x, dropout_prob)]

        net += [ReflectionPad(pad_width=1),
                EqualizedConv2d(middle_channels,
                                out_channels,
                                ksize=4,
                                pad=0,
                                stride=2,
                                nobias=False if norm is None else True)]

        if norm is not None:
            net += [norm(out_channels)]

        net += [F.leaky_relu]

        if use_dropout:
            net += [lambda x: F.dropout(x, dropout_prob)]

        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Unet3d(chainer.Chain):
    '''A 3d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 num_down,
                 max_channels,
                 norm=L.BatchNormalization,
                 outermost_linear=False):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet3d."

        # Define the in block
        in_layer = [Conv3dSame(in_channels, nf0, kernel_size=3, bias=False)]

        if norm is not None:
            in_layer += [norm(nf0)]

        in_layer += [F.leaky_relu]

        # Define the center UNet block. The feature map has height and width 1 --> no batchnorm.
        unet_block = UnetSkipConnectionBlock3d(int(min(2 ** (num_down - 1) * nf0, max_channels)),
                                               int(min(2 ** (num_down - 1) * nf0, max_channels)),
                                               norm=None)
        for i in list(range(0, num_down - 1))[::-1]:
            unet_block = UnetSkipConnectionBlock3d(int(min(2 ** i * nf0, max_channels)),
                                                   int(min(2 ** (i + 1) * nf0, max_channels)),
                                                   submodule=unet_block,
                                                   norm=norm)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        out_layer = [Conv3dSame(2 * nf0,
                                out_channels,
                                kernel_size=3,
                                bias=outermost_linear)]

        if not outermost_linear:
            if norm is not None:
                out_layer += [norm(out_channels)]
            out_layer += [F.relu]
        with self.init_scope():
            self.in_layer = chainer.Sequential(*in_layer)
            self.unet_block = unet_block
            self.out_layer = chainer.Sequential(*out_layer)

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer


def interpolate_trilinear(grid, lin_ind_frustrum, voxel_coords, img_shape, frustrum_depth):
    xp = chainer.cuda.get_array_module(voxel_coords)
    batch, num_feats, height, width, depth = grid.shape

    lin_ind_frustrum = lin_ind_frustrum.astype("int32")  # indexを指定するだけなので勾配を流す必要はない

    x_indices = voxel_coords[2, :]
    y_indices = voxel_coords[1, :]
    z_indices = voxel_coords[0, :]

    x0 = x_indices.astype("int32")
    y0 = y_indices.astype("int32")
    z0 = z_indices.astype("int32")

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x1 = xp.clip(x1, 0, width - 1)
    y1 = xp.clip(y1, 0, height - 1)
    z1 = xp.clip(z1, 0, depth - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    # output = torch.zeros(batch, num_feats, img_shape[0]*img_shape[1]*depth).cuda()
    output = xp.zeros((batch, num_feats, img_shape[0] * img_shape[1] * frustrum_depth), dtype="float32")
    added = grid[:, :, x0, y0, z0] * (1 - x) * (1 - y) * (1 - z) + \
            grid[:, :, x1, y0, z0] * x * (1 - y) * (1 - z) + \
            grid[:, :, x0, y1, z0] * (1 - x) * y * (1 - z) + \
            grid[:, :, x0, y0, z1] * (1 - x) * (1 - y) * z + \
            grid[:, :, x1, y0, z1] * x * (1 - y) * z + \
            grid[:, :, x0, y1, z1] * (1 - x) * y * z + \
            grid[:, :, x1, y1, z0] * x * y * (1 - z) + \
            grid[:, :, x1, y1, z1] * x * y * z
    make_slice = MakeSlice()
    output = F.scatter_add(output, make_slice[:, :, lin_ind_frustrum], added)

    output = output.reshape(batch, num_feats, frustrum_depth, img_shape[0], img_shape[1])
    return output


class MakeSlice:
    def __getitem__(self, item):
        return item


def num_divisible_by_2(number):
    i = 0
    while not number % 2:
        number = number // 2
        i += 1

    return i


# class IntegrationNet(chainer.Chain):
#     '''The 3D integration net integrating new observations into the Deepvoxels grid.
#     '''
#
#     def __init__(self, nf0, coord_conv, use_dropout, per_feature, grid_dim):
#         super().__init__()
#
#         self.coord_conv = coord_conv
#         if self.coord_conv:
#             in_channels = nf0 + 3
#         else:
#             in_channels = nf0
#
#         if per_feature:
#             weights_channels = nf0
#         else:
#             weights_channels = 1
#
#         self.use_dropout = use_dropout
#         with self.init_scope():
#             self.new_integration = chainer.Sequential(
#                 ReplicationPad(1),
#                 L.Convolution3D(in_channels, nf0, ksize=3, pad=0, nobias=False),
#                 lambda x: F.dropout(x, 0.2)
#             )
#
#             self.old_integration = chainer.Sequential(
#                 ReplicationPad(1),
#                 L.Convolution3D(in_channels, nf0, ksize=3, pad=0, nobias=False),
#                 lambda x: F.dropout(x, 0.2)
#             )
#
#             self.update_old_net = chainer.Sequential(
#                 ReplicationPad(1),
#                 L.Convolution3D(in_channels, weights_channels, ksize=3, pad=0, nobias=False),
#             )
#             self.update_new_net = chainer.Sequential(
#                 ReplicationPad(1),
#                 L.Convolution3D(in_channels, weights_channels, ksize=3, pad=0, nobias=True),
#             )
#
#             self.reset_old_net = chainer.Sequential(
#                 ReplicationPad(1),
#                 L.Convolution3D(in_channels, weights_channels, ksize=3, pad=0, nobias=False),
#             )
#             self.reset_new_net = chainer.Sequential(
#                 ReplicationPad(1),
#                 L.Convolution3D(in_channels, weights_channels, ksize=3, pad=0, nobias=True),
#             )
#
#         self.sigmoid = F.sigmoid
#         self.relu = F.relu
#
#         coord_conv_volume = np.mgrid[-grid_dim // 2:grid_dim // 2,
#                             -grid_dim // 2:grid_dim // 2,
#                             -grid_dim // 2:grid_dim // 2]
#
#         coord_conv_volume = np.stack(coord_conv_volume, axis=0).astype(np.float32)[None, :, :, :, :]
#         self.coord_conv_volume = coord_conv_volume / grid_dim
#         self.register_persistent('coord_conv_volume')
#         self.counter = 0
#
#     def forward(self, new_observation, old_state, writer):
#
#         old_state_coord = F.concat([old_state, self.coord_conv_volume], axis=1)
#         new_observation_coord = F.concat([new_observation, self.coord_conv_volume], axis=1)
#
#         reset = self.sigmoid(self.reset_old_net(old_state_coord) + self.reset_new_net(new_observation_coord))
#         update = self.sigmoid(self.update_old_net(old_state_coord) + self.update_new_net(new_observation_coord))
#
#         final = self.relu(self.new_integration(new_observation_coord) + self.old_integration(reset * old_state_coord))
#
#         if not self.counter % 100:
#             # Plot the volumes
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             update_values = update.mean(dim=1).squeeze().cpu().detach().numpy()
#             x, y, z = np.where(update_values)
#             x, y, z = x[::3], y[::3], z[::3]
#             ax.scatter(x, y, z, s=update_values[x, y, z] * 5)
#
#             writer.add_figure("update_gate",
#                               fig,
#                               self.counter,
#                               close=True)
#
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             reset_values = reset.mean(dim=1).squeeze().cpu().detach().numpy()
#             x, y, z = np.where(reset_values)
#             x, y, z = x[::3], y[::3], z[::3]
#             ax.scatter(x, y, z, s=reset_values[x, y, z] * 5)
#             writer.add_figure("reset_gate",
#                               fig,
#                               self.counter,
#                               close=True)
#         self.counter += 1
#
#         result = ((1 - update) * old_state + update * final)
#         return result


class OcclusionNet(chainer.Chain):
    '''The Occlusion Module predicts visibility scores for each voxel across a ray, allowing occlusion reasoning
    via a convex combination of voxels along each ray.
    '''

    def __init__(self, nf0, occnet_nf, frustrum_dims):
        super().__init__()

        self.occnet_nf = occnet_nf
        self.frustrum_dims = frustrum_dims
        self.frustrum_depth = frustrum_dims[-1]
        self.depth_coords = None

        with self.init_scope():
            self.occlusion_prep = chainer.Sequential(
                Conv3dSame(nf0 + 1, self.occnet_nf, kernel_size=3, bias=False),
                L.BatchNormalization(self.occnet_nf),
                F.relu,
            )

            num_down = min(num_divisible_by_2(self.frustrum_depth),
                           num_divisible_by_2(frustrum_dims[0]))

            self.occlusion_net = Unet3d(in_channels=self.occnet_nf,
                                        out_channels=self.occnet_nf,
                                        nf0=self.occnet_nf,
                                        num_down=num_down,
                                        max_channels=4 * self.occnet_nf,
                                        outermost_linear=False)

            self.softmax_net = chainer.Sequential(
                Conv3dSame(2 * self.occnet_nf + 1, 1, kernel_size=3, bias=True),
                lambda x: F.softmax(x, axis=2),
            )
        depth_coords = np.arange(-self.frustrum_depth // 2,
                                 self.frustrum_depth // 2)[None, None, :, None, None] / self.frustrum_depth
        self.depth_coords = np.tile(depth_coords,
                                    (1, 1, 1, self.frustrum_dims[0], self.frustrum_dims[0])).astype("float32")
        self.register_persistent('depth_coords')

    def forward(self, novel_img_frustrum):
        frustrum_feats_depth = F.concat([self.depth_coords, novel_img_frustrum], axis=1)

        occlusion_prep = self.occlusion_prep(frustrum_feats_depth)
        frustrum_feats = self.occlusion_net(occlusion_prep)
        frustrum_weights = self.softmax_net(F.concat([occlusion_prep, frustrum_feats, self.depth_coords], axis=1))

        depth_map = F.sum(self.depth_coords * frustrum_weights, axis=2)  # -0.5 ~ 0.5

        return frustrum_weights, depth_map


class OcclusionNetLight(chainer.Chain):
    '''The Occlusion Module predicts visibility scores for each voxel across a ray, allowing occlusion reasoning
    via a convex combination of voxels along each ray.
    '''

    def __init__(self, nf0, occnet_nf, frustrum_dims):
        super().__init__()

        self.occnet_nf = occnet_nf
        self.frustrum_dims = frustrum_dims
        self.frustrum_depth = frustrum_dims[-1]
        self.depth_coords = None

        with self.init_scope():
            self.occlusion_prep = chainer.Sequential(
                Conv3dSame(nf0 + 1, self.occnet_nf, kernel_size=3, bias=False),
                L.BatchNormalization(self.occnet_nf),
                F.leaky_relu,
            )

            self.occlusion_net = chainer.Sequential(
                Conv3dSame(self.occnet_nf, self.occnet_nf, kernel_size=3, bias=False),
                L.BatchNormalization(self.occnet_nf),
                F.leaky_relu,
            )

            self.softmax_net = chainer.Sequential(
                Conv3dSame(2 * self.occnet_nf + 1, 1, kernel_size=3, bias=True),
                lambda x: F.softmax(x, axis=2),
            )
        depth_coords = np.arange(-self.frustrum_depth // 2,
                                 self.frustrum_depth // 2)[None, None, :, None, None] / self.frustrum_depth
        self.depth_coords = np.tile(depth_coords,
                                    (1, 1, 1, self.frustrum_dims[0], self.frustrum_dims[0])).astype("float32")
        self.register_persistent('depth_coords')

    def forward(self, novel_img_frustrum):
        frustrum_feats_depth = F.concat([self.depth_coords, novel_img_frustrum], axis=1)

        occlusion_prep = self.occlusion_prep(frustrum_feats_depth)
        frustrum_feats = self.occlusion_net(occlusion_prep)
        frustrum_weights = self.softmax_net(F.concat([occlusion_prep, frustrum_feats, self.depth_coords], axis=1))
        depth_map = F.sum(self.depth_coords * frustrum_weights, axis=2)  # -0.5 ~ 0.5

        return frustrum_weights, depth_map


class AccumulativeOcclusionNet(chainer.Chain):
    '''The Occlusion Module predicts visibility scores for each voxel across a ray, allowing occlusion reasoning
    via a convex combination of voxels along each ray.
    '''

    def __init__(self, nf0, occnet_nf, frustrum_dims, accmulative_threshold=None):
        super().__init__()

        self.occnet_nf = occnet_nf
        self.frustrum_dims = frustrum_dims
        self.frustrum_depth = frustrum_dims[-1]
        self.depth_coords = None
        self.accmulative_threshold = accmulative_threshold if accmulative_threshold else 4
        print(self.accmulative_threshold)

        with self.init_scope():
            self.occlusion = chainer.Sequential(
                Conv3dSame(nf0 + 1, self.occnet_nf, kernel_size=1, bias=True),
                # L.BatchNormalization(self.occnet_nf),
                F.leaky_relu,
                Conv3dSame(self.occnet_nf, 1, kernel_size=1, bias=True),
                lambda x: x - self.accmulative_threshold,
                F.sigmoid,
            )
        depth_coords = np.arange(-self.frustrum_depth // 2,
                                 self.frustrum_depth // 2)[None, None, :, None, None] / self.frustrum_depth
        self.depth_coords = np.tile(depth_coords,
                                    (1, 1, 1, self.frustrum_dims[0], self.frustrum_dims[0])).astype("float32")
        self.register_persistent('depth_coords')

    def forward(self, novel_img_frustrum):
        frustrum_feats_depth = F.concat([self.depth_coords, novel_img_frustrum], axis=1)
        occlusion_prep = self.occlusion(frustrum_feats_depth)
        # print(occlusion_prep.array)
        b, c, d, h, w = occlusion_prep.shape
        cumsum = F.concat([self.xp.zeros((b, c, 1, h, w), "float32"),
                           F.clip(F.cumsum(occlusion_prep, axis=2), 0, 1)], axis=2)
        # print(cumsum[0, 0, :, 0, 0])
        frustum_weights = cumsum[:, :, 1:] - cumsum[:, :, :-1]
        depth_map = F.sum(self.depth_coords * frustum_weights, axis=2)  # -0.5 ~ 0.5

        # print(depth_map.array.mean())

        return frustum_weights, depth_map


class RenderNetProjection(chainer.Chain):
    '''The Occlusion Module predicts visibility scores for each voxel across a ray, allowing occlusion reasoning
    via a convex combination of voxels along each ray.
    '''

    def __init__(self, nf0, occnet_nf, frustrum_dims):
        super().__init__()

        self.occnet_nf = 32  # TODO avoid hard coding
        self.frustrum_dims = frustrum_dims
        self.frustrum_depth = frustrum_dims[-1]
        self.depth_coords = None

        with self.init_scope():
            self.mlp = chainer.Sequential(
                EqualizedConv2d(nf0 * self.frustrum_depth, self.occnet_nf, 1, 1, 0),
                L.BatchNormalization(self.occnet_nf),
                F.leaky_relu,
                EqualizedConv2d(self.occnet_nf, self.occnet_nf),
                L.BatchNormalization(self.occnet_nf),
                F.leaky_relu,
            )

    def forward(self, novel_img_frustrum):
        b, c, d, h, w = novel_img_frustrum.shape
        novel_img_frustrum = novel_img_frustrum.reshape(b, c * d, h, w)

        projected_feats = self.mlp(novel_img_frustrum)  # b x ? x h x w
        return projected_feats


class UnetSkipConnectionBlock(chainer.Chain):
    '''Helper class for building a 2D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 upsampling_mode,
                 norm=L.BatchNormalization,
                 submodule=None,
                 use_dropout=False,
                 dropout_prob=0.1):
        super().__init__()

        if submodule is None:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     UpBlock(inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]
        else:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     submodule,
                     UpBlock(2 * inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]
        with self.init_scope():
            self.model = chainer.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return F.concat([x, forward_passed], 1)


class Unet(chainer.Chain):
    '''A 2d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 num_down,
                 max_channels,
                 use_dropout,
                 upsampling_mode='transpose',
                 dropout_prob=0.1,
                 norm=L.BatchNormalization,
                 outermost_linear=False):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param use_dropout: Whether to use dropout or no.
        :param dropout_prob: Dropout probability if use_dropout=True.
        :param upsampling_mode: Which type of upsampling should be used. See "UpBlock" for documentation.
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet."

        # Define the in block
        in_layer = [Conv2dSame(in_channels, nf0, kernel_size=3, bias=True if norm is None else False)]
        if norm is not None:
            in_layer += [norm(nf0)]
        in_layer += [F.leaky_relu]

        if use_dropout:
            in_layer += [lambda x: F.dropout(x, dropout_prob)]

        # Define the center UNet block
        unet_block = UnetSkipConnectionBlock(min(2 ** (num_down - 1) * nf0, max_channels),
                                             min(2 ** (num_down - 1) * nf0, max_channels),
                                             use_dropout=use_dropout,
                                             dropout_prob=dropout_prob,
                                             norm=None,  # Innermost has no norm (spatial dimension 1)
                                             upsampling_mode=upsampling_mode)

        for i in list(range(0, num_down - 1))[::-1]:
            unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),
                                                 min(2 ** (i + 1) * nf0, max_channels),
                                                 use_dropout=use_dropout,
                                                 dropout_prob=dropout_prob,
                                                 submodule=unet_block,
                                                 norm=norm,
                                                 upsampling_mode=upsampling_mode)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        out_layer = [Conv2dSame(2 * nf0,
                                out_channels,
                                kernel_size=3,
                                bias=outermost_linear or (norm is None))]

        if not outermost_linear:
            if norm is not None:
                out_layer += [norm(out_channels)]
            out_layer += [F.relu]

            if use_dropout:
                out_layer += [lambda x: F.dropout(x, dropout_prob)]
        with self.init_scope():
            self.in_layer = chainer.Sequential(*in_layer)
            self.unet_block = unet_block
            self.out_layer = chainer.Sequential(*out_layer)

        # self.out_layer_weight = self.out_layer[0].weight

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer


class Identity:
    '''Helper module to allow Downsampling and Upsampling nets to default to identity if they receive an empty list.'''

    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return input


class UpsamplingNet(chainer.Chain):
    '''A subnetwork that upsamples a 2D feature map with a variety of upsampling options.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 upsampling_mode,
                 use_dropout,
                 dropout_prob=0.1,
                 first_layer_one=False,
                 norm=L.BatchNormalization):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of upsampling steps (each step upsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param upsampling_mode: Mode of upsampling. For documentation, see class "UpBlock"
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param first_layer_one: Whether the input to the last layer will have a spatial size of 1. In that case,
                               the first layer will not have a norm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()

        if not len(per_layer_out_ch):
            self.ups = Identity()
        else:
            ups = list()
            ups.append(UpBlock(in_channels,
                               per_layer_out_ch[0],
                               use_dropout=use_dropout,
                               dropout_prob=dropout_prob,
                               norm=None if first_layer_one else norm,
                               upsampling_mode=upsampling_mode))
            for i in range(0, len(per_layer_out_ch) - 1):
                ups.append(
                    UpBlock(per_layer_out_ch[i],
                            per_layer_out_ch[i + 1],
                            use_dropout=use_dropout,
                            dropout_prob=dropout_prob,
                            norm=norm,
                            upsampling_mode=upsampling_mode))
            with self.init_scope():
                self.ups = chainer.Sequential(*ups)

    def forward(self, input):
        return self.ups(input)


class RenderingNet(chainer.Chain):
    def __init__(self,
                 nf0,
                 in_channels,
                 input_resolution,
                 img_sidelength):
        super().__init__()

        num_down_unet = num_divisible_by_2(input_resolution)
        num_upsampling = num_divisible_by_2(img_sidelength) - num_down_unet

        net = [
            Unet(in_channels=in_channels,
                 out_channels=3 if num_upsampling <= 0 else 4 * nf0,
                 outermost_linear=True if num_upsampling <= 0 else False,
                 use_dropout=True,
                 dropout_prob=0.1,
                 nf0=nf0 * (2 ** num_upsampling),
                 norm=L.BatchNormalization,
                 max_channels=8 * nf0,
                 num_down=num_down_unet)
        ]

        if num_upsampling > 0:
            net += [
                UpsamplingNet(per_layer_out_ch=num_upsampling * [nf0],
                              in_channels=4 * nf0,
                              upsampling_mode='transpose',
                              use_dropout=True,
                              dropout_prob=0.1),
                Conv2dSame(nf0, out_channels=nf0 // 2, kernel_size=3, bias=False),
                L.BatchNormalization(nf0 // 2),
                F.relu,
                Conv2dSame(nf0 // 2, 3, kernel_size=3)
            ]

        # net += [F.tanh]
        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, input):
        return self.net(input)


class RenderingNetLight(chainer.Chain):
    def __init__(self,
                 nf0,
                 in_channels,
                 input_resolution,
                 img_sidelength):
        super().__init__()

        num_down_unet = num_divisible_by_2(input_resolution)
        num_upsampling = num_divisible_by_2(img_sidelength) - num_down_unet
        assert num_upsampling <= 0

        net = [
            Conv2dSame(in_channels, in_channels, kernel_size=3, bias=False),
            L.BatchNormalization(in_channels),
            F.relu,
            Conv2dSame(in_channels, 3 if num_upsampling <= 0 else 4 * nf0, kernel_size=3, bias=False)
        ]

        if num_upsampling > 0:
            net += [
                UpsamplingNet(per_layer_out_ch=num_upsampling * [nf0],
                              in_channels=4 * nf0,
                              upsampling_mode='transpose',
                              use_dropout=True,
                              dropout_prob=0.1),
                Conv2dSame(nf0, out_channels=nf0 // 2, kernel_size=3, bias=False),
                L.BatchNormalization(nf0 // 2),
                F.relu,
                Conv2dSame(nf0 // 2, 3, kernel_size=3)
            ]

        # net += [F.tanh]
        with self.init_scope():
            self.net = chainer.Sequential(*net)

    def forward(self, input):
        return self.net(input)


class DeepVoxels(chainer.Chain):
    def __init__(self,
                 img_sidelength,
                 lifting_img_dims,
                 frustrum_img_dims,
                 grid_dims,
                 num_grid_feats=64,
                 nf0=64,
                 occlusion_type="deepvoxels",
                 voxel_size=0,
                 near_plane=0,
                 config=None):
        ''' Initializes the DeepVoxels model.

        :param img_sidelength: The sidelength of the input images (for instance 512)
        :param lifting_img_dims: The dimensions of the feature map to be lifted.
        :param frustrum_img_dims: The dimensions of the canonical view volume that DeepVoxels are resampled to.
        :param grid_dims: The dimensions of the deepvoxels grid.
        :param grid_dims: The number of featres in the outermost layer of U-Nets.
        :param use_occlusion_net: Whether to use the OcclusionNet or not.
        '''
        super().__init__()

        self.occlusion_type = occlusion_type
        self.grid_dims = grid_dims

        self.norm = L.BatchNormalization

        self.lifting_img_dims = lifting_img_dims
        self.frustrum_img_dims = frustrum_img_dims
        self.grid_dims = grid_dims

        # The frustrum depth is the number of voxels in the depth dimension of the canonical viewing volume.
        # It's calculated as the length of the diagonal of the DeepVoxels grid.
        self.frustrum_depth = int(np.ceil(np.sqrt(3) * grid_dims[-1]))

        self.nf0 = nf0  # Number of features to use in the outermost layer of all U-Nets
        self.n_grid_feats = num_grid_feats  # Number of features in the DeepVoxels grid.
        self.occnet_nf = 4  # Number of features to use in the 3D unet of the occlusion subnetwork

        self.voxel_size = voxel_size
        self.near_plane = near_plane

        with self.init_scope():
            # Rendering net is an asymmetric UNet: UNet with skip connections and then straight upsampling
            # self.rendering_net = RenderingNetLight(nf0=self.nf0,
            #                                        in_channels=self.n_grid_feats,
            #                                        input_resolution=self.frustrum_img_dims[0],
            #                                        img_sidelength=img_sidelength)
            if self.occlusion_type == "deepvoxels":
                self.occlusion_net = OcclusionNetLight(nf0=self.n_grid_feats,
                                                       occnet_nf=self.occnet_nf,
                                                       frustrum_dims=[self.frustrum_img_dims[0],
                                                                      self.frustrum_img_dims[1],
                                                                      self.frustrum_depth])
            elif self.occlusion_type == "accumulative":
                self.occlusion_net = AccumulativeOcclusionNet(nf0=self.n_grid_feats,
                                                              occnet_nf=self.occnet_nf,
                                                              frustrum_dims=[self.frustrum_img_dims[0],
                                                                             self.frustrum_img_dims[1],
                                                                             self.frustrum_depth],
                                                              accmulative_threshold=config.accumulative_threshold)
            elif self.occlusion_type == "rendernet":
                self.occlusion_net = RenderNetProjection(nf0=self.n_grid_feats,
                                                         occnet_nf=self.occnet_nf,
                                                         frustrum_dims=[self.frustrum_img_dims[0],
                                                                        self.frustrum_img_dims[1],
                                                                        self.frustrum_depth])
            else:
                assert False

            # self.integration_net = IntegrationNet(self.n_grid_feats,
            #                                       use_dropout=True,
            #                                       coord_conv=True,
            #                                       per_feature=False,
            #                                       grid_dim=grid_dims[-1])

            # self.inpainting_net = Unet3d(in_channels=self.n_grid_feats + 3,
            #                              out_channels=self.n_grid_feats,
            #                              num_down=2,
            #                              nf0=self.n_grid_feats,
            #                              max_channels=4 * self.n_grid_feats)

        # print(100 * "*")
        # # print("inpainting_net")
        # # util.print_network(self.inpainting_net)
        # # print(self.inpainting_net)
        # print("rendering net")
        # # util.print_network(self.rendering_net)
        # print(self.rendering_net)
        # print("feature extraction net")
        # # util.print_network(self.feature_extractor)
        # # print(self.feature_extractor)
        # print(100 * "*")

        # Coordconv volumes
        coord_conv_volume = np.mgrid[-self.grid_dims[0] // 2:self.grid_dims[0] // 2,
                            -self.grid_dims[1] // 2:self.grid_dims[1] // 2,
                            -self.grid_dims[2] // 2:self.grid_dims[2] // 2]

        coord_conv_volume = np.stack(coord_conv_volume, axis=0)[None, :, :, :, :]
        self.coord_conv_volume = (coord_conv_volume / self.grid_dims[0]).astype(np.float32)
        self.register_persistent('coord_conv_volume')

    def forward(self, proj_frustrum_idcs_list, proj_grid_coords_list, deepvoxels, return_foreground_weight=False):
        # deepvoxelがすでに生成されているとして
        # dv_new = deepvoxels

        # inpainting_input = F.concat([dv_new,
        #                              F.tile(self.coord_conv_volume, (dv_new.shape[0], 1, 1, 1, 1))],
        #                             axis=1)  # coord conv
        # dv_inpainted = self.inpainting_net(inpainting_input)  # unet
        dv_inpainted = deepvoxels  # omit inpainting_net
        novel_views, depth_maps = list(), list()
        if return_foreground_weight:
            assert self.occlusion_type in ["deepvoxels", "accumulative"], "invalid occlusion type to use background"
            foreground_weight = list()

        for i, (proj_frustrum_idcs, proj_grid_coords) in enumerate(zip(proj_frustrum_idcs_list, proj_grid_coords_list)):
            can_view_vol = interpolate_trilinear(dv_inpainted[None, i],
                                                 proj_frustrum_idcs,
                                                 proj_grid_coords,
                                                 self.frustrum_img_dims,
                                                 self.frustrum_depth)
            if self.occlusion_type in ["deepvoxels", "accumulative"]:
                visibility_weights, depth_map = self.occlusion_net(can_view_vol)  # frustumをCNNに突っ込んでsoftmaxかけてるだけ
                depth_maps.append(depth_map)
                collapsed_frustrum = F.sum(visibility_weights * can_view_vol, axis=2)
                novel_image_features = collapsed_frustrum.reshape(1, -1, self.frustrum_img_dims[0],
                                                                  self.frustrum_img_dims[1])
                if return_foreground_weight:
                    foreground_weight.append(F.sum(visibility_weights, axis=2))
            else:
                novel_image_features = self.occlusion_net(can_view_vol)
                b, c, h, w = novel_image_features.shape
                depth_maps.append(self.xp.ones((b, 1, h, w), dtype="float32"))

            # rendered_img = 0.5 * self.rendering_net(novel_image_features)
            novel_views.append(novel_image_features)
        novel_views = F.concat(novel_views, axis=0)
        depth_maps = F.concat(depth_maps, axis=0)

        depth_maps = ((depth_maps + 0.5) * int(np.ceil(np.sqrt(3) * self.grid_dims[-1])) *
                      self.voxel_size + self.near_plane)  # fruxtrum_depth = int(np.ceil(np.sqrt(3) * grid_dims[-1])なので
        if return_foreground_weight:
            foreground_weight = F.concat(foreground_weight, axis=0)
            return novel_views, depth_maps, foreground_weight

        return novel_views, depth_maps


def train():
    print('Begin training...')
    trgt_views = [xp.array([[-9.96363e-01, -1.47920e-02, 8.39167e-02, -1.25875e-01],
                            [1.86265e-09, 9.84817e-01, 1.73594e-01, -2.60391e-01],
                            [8.52104e-02, -1.72963e-01, 9.81236e-01, -1.47185e+00],
                            [0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00]], dtype="float32")]
    # trgt_views, nearest_view = dataloader.__next__()
    # backproj_mapping = projection.comp_lifting_idcs(camera_to_world=nearest_view['pose'].squeeze().to(device),
    #                                                 grid2world=grid_origin)
    deepvoxels = xp.zeros((1, 64, 32, 32, 32), dtype="float32")

    proj_mappings = list()
    for i in range(len(trgt_views)):
        proj_mappings.append(projection.compute_proj_idcs(trgt_views[i],
                                                          grid2world=grid_origin))

    # lift_volume_idcs, lift_img_coords = backproj_mapping

    proj_frustrum_idcs, proj_grid_coords = list(zip(*proj_mappings))

    outputs, depth_maps = model(proj_frustrum_idcs, proj_grid_coords, deepvoxels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--train_test', type=str, required=True,
    #                     help='Whether to run training or testing. Options are \"train\" or \"test\".')
    # parser.add_argument('--data_root', required=True,
    #                     help='Path to directory that holds the object data. See dataio.py for directory structure etc..')
    # parser.add_argument('--logging_root', required=True,
    #                     help='Path to directory where to write tensorboard logs and checkpoints.')

    parser.add_argument('--experiment_name', type=str, default='', help='(optional) Name for experiment.')
    parser.add_argument('--max_epoch', type=int, default=400, help='Maximum number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate.')
    parser.add_argument('--l1_weight', type=float, default=200, help='Weight of l1 loss.')
    parser.add_argument('--sampling_pattern', type=str, default='skip_2', required=False,
                        help='Whether to use \"all\" images or whether to skip n images (\"skip_1\" picks every 2nd image.')

    parser.add_argument('--img_sidelength', type=int, default=512,
                        help='Sidelength of generated images. Default 512. Only less than native resolution of images is recommended.')

    parser.add_argument('--no_occlusion_net', action='store_true', default=False,
                        help='Disables occlusion net and replaces it with a fully convolutional 2d net.')
    parser.add_argument('--num_trgt', type=int, default=2, required=False,
                        help='How many novel views will be generated at training time.')

    parser.add_argument('--checkpoint', default='',
                        help='Path to a checkpoint to load model weights from.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Start epoch')

    parser.add_argument('--grid_dim', type=int, default=32,
                        help='Grid sidelength. Default 32.')
    parser.add_argument('--num_grid_feats', type=int, default=64,
                        help='Number of features stored in each voxel.')
    parser.add_argument('--nf0', type=int, default=64,
                        help='Number of features in outermost layer of U-Net architectures.')
    parser.add_argument('--near_plane', type=float, default=np.sqrt(3) / 2,
                        help='Position of the near plane.')

    opt = parser.parse_args()
    print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    cuda.get_device_from_id(opt.gpu).use()
    xp = cuda.cupy

    input_image_dims = [opt.img_sidelength, opt.img_sidelength]
    proj_image_dims = [64, 64]  # Height, width of 2d feature map used for lifting and rendering.

    # Read origin of grid, scale of each voxel, and near plane
    # _, grid_barycenter, scale, near_plane, _ = \
    #     util.parse_intrinsics(os.path.join(opt.data_root, 'intrinsics.txt'), trgt_sidelength=input_image_dims[0])

    grid_barycenter = xp.array([0, 0, 0], "float32")
    scale = 1.
    near_plane = 0.

    if near_plane == 0.0:
        near_plane = opt.near_plane

    # Read intrinsic matrix for lifting and projection
    # lift_intrinsic = util.parse_intrinsics(os.path.join(opt.data_root, 'intrinsics.txt'),
    #                                        trgt_sidelength=proj_image_dims[0])[0]
    lift_intrinsic = np.array([[560, 0.0, 256, 0.0],
                               [0.0, 560, 256, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    proj_intrinsic = lift_intrinsic

    # Set up scale and world coordinates of voxel grid
    voxel_size = (1. / opt.grid_dim) * 1.1 * scale
    grid_origin = xp.eye(4, dtype="float32")
    grid_origin[:3, 3] = grid_barycenter  # ただの単位行列

    # Minimum and maximum depth used for rejecting voxels outside of the cmaera frustrum
    depth_min = 0.
    depth_max = opt.grid_dim * voxel_size + near_plane
    grid_dims = 3 * [opt.grid_dim]

    # Resolution of canonical viewing volume in the depth dimension, in number of voxels.
    frustrum_depth = int(np.ceil(np.sqrt(3) * grid_dims[-1]))

    model = DeepVoxels(lifting_img_dims=proj_image_dims,
                       frustrum_img_dims=proj_image_dims,
                       grid_dims=grid_dims,
                       use_occlusion_net=not opt.no_occlusion_net,
                       num_grid_feats=opt.num_grid_feats,
                       nf0=opt.nf0,
                       img_sidelength=input_image_dims[0]).to_gpu()

    # Projection module
    projection = ProjectionHelper(projection_intrinsic=proj_intrinsic,
                                  lifting_intrinsic=lift_intrinsic,
                                  depth_min=depth_min,
                                  depth_max=depth_max,
                                  projection_image_dims=proj_image_dims,
                                  lifting_image_dims=proj_image_dims,
                                  grid_dims=grid_dims,
                                  voxel_size=voxel_size,
                                  device=None,
                                  frustrum_depth=frustrum_depth,
                                  near_plane=near_plane)

    train()
