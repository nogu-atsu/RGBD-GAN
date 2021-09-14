import chainer
import chainer.functions as F
import numpy as np
from chainer import Variable


def loss_l2(h, t):
    return F.sum((h - t) ** 2) / np.prod(h.data.shape)


def loss_l2_no_avg(h, t):
    return F.sum((h - t) ** 2) / np.prod(h.data.shape[1:])


def loss_func_dcgan_gen(y_fake, focal_loss_gamma=0.):
    if focal_loss_gamma is None:
        focal_loss_gamma = 0.
    return F.sum(F.softplus(-y_fake) * F.sigmoid(-y_fake) ** focal_loss_gamma) / np.prod(y_fake.data.shape)


def loss_func_dcgan_dis(y_fake, y_real):
    if isinstance(y_fake, tuple):
        loss = 0
        for _y_fake, _y_real in zip(y_fake, y_real):
            loss += F.sum(F.softplus(_y_fake)) / np.prod(_y_fake.data.shape)
            loss += F.sum(F.softplus(-_y_real)) / np.prod(_y_real.data.shape)

    else:
        loss = F.sum(F.softplus(y_fake)) / np.prod(y_fake.data.shape)
        loss += F.sum(F.softplus(-y_real)) / np.prod(y_real.data.shape)

    return loss


class LossFuncRotate:
    def __init__(self, xp, K=None, norm="l1", lambda_geometric=3):
        self.xp = xp
        self.size = None
        self.K = K
        self.norm = norm
        self.lambda_geometric = lambda_geometric

    def init_params(self, xp, size=4):
        # camera intrinsic parameter
        if self.size is None:
            if self.K is not None:
                self.K = xp.array(self.K[:3, :3], "float32")
                self.K[:2] *= size / self.K[0, 2] / 2
                self.size = size
            else:
                self.size = size
                self.K = xp.array([[size * 2, 0, size / 2],
                                   [0, size * 2, size / 2],
                                   [0, 0, 1]], dtype="float32")

        else:
            self.size = size
            self.K[:2] *= size / self.K[0, 2] / 2

        self.inv_K = xp.linalg.inv(self.K).astype("float32")
        # position in image
        # 画像の横, 縦, 奥行きの順
        self.p = xp.asarray(
            xp.meshgrid(xp.arange(size), xp.arange(size)) + [xp.ones((size, size))], dtype="float32"
        ).reshape(3, -1)

    def __call__(self, img, theta, img_rot, theta_rot, occlusion_aware=False, debug=False, max_depth=None,
                 min_depth=None):
        '''
        l1 loss between img and rotated img
        :param img: b x 3 x h x w
        :param depth: b x 1 x h x w
        :param img_rot: b x 3 x h x w
        :param depth_rot: b x 1 x h x w
        :return: l1 loss between color (and depth??)
        '''
        xp = self.xp
        if self.size != img.shape[-1]:
            self.init_params(xp, size=img.shape[-1])
        z = img[:, -1:]  # b x 1 x h x w
        z_rot = img_rot[:, -1:]

        z = z.reshape(z.shape[0], 1, -1)  # b x 1 x hw
        z_rot = z_rot.reshape(z_rot.shape[0], 1, -1)

        if isinstance(theta, Variable):
            theta = theta.array
            theta_rot = theta_rot.array
        R1 = theta[:, :3, :3]
        R2 = theta_rot[:, :3, :3]
        t1 = theta[:, :3, -1:]
        t2 = theta_rot[:, :3, -1:]
        R = xp.matmul(R2.transpose(0, 2, 1), R1).astype("float32")
        inv_R = R.transpose(0, 2, 1)
        t = xp.matmul(R1.transpose(0, 2, 1), t2 - t1).astype("float32")

        new_zp = warp(self.K, self.inv_K, R, t, z, self.p)
        new_zp_rot = inv_warp(self.K, self.inv_K, inv_R, t, z_rot, self.p)

        # 元の画像の色と，移動先での画像の色の距離
        # はみ出したところには勾配を流さない
        warped, not_out = bilinear(img_rot, new_zp)
        warped_rot, not_out_rot = bilinear(img, new_zp_rot)
        if debug:
            # just output warped images for debugging
            return warped, not_out, new_zp, warped_rot, not_out_rot, new_zp_rot

        # calculate reversed surface
        # calculate normal map

        warped_target = F.concat([img[:, :-1].transpose(0, 2, 3, 1).reshape(-1, img.shape[1] - 1),
                                  new_zp[:, :, 2].reshape(-1, 1)]) * not_out[:, None]
        warped_rot_target = F.concat([img_rot[:, :-1].transpose(0, 2, 3, 1).reshape(-1, img.shape[1] - 1),
                                      new_zp_rot[:, :, 2].reshape(-1, 1)]) * not_out_rot[:, None]

        if occlusion_aware:
            # calculate occlusion
            not_occluded = warped[:, -1:].array > new_zp[:, :, 2].reshape(-1, 1).array
            not_occluded_rot = warped_rot[:, -1:].array > new_zp_rot[:, :, 2].reshape(-1, 1).array
            warped = warped * not_occluded
            warped_rot = warped_rot * not_occluded_rot
            warped_target = warped_target * not_occluded
            warped_rot_target = warped_rot_target * not_occluded_rot

        if max_depth is not None:
            small_depth = z.array.transpose(0, 2, 1).reshape(-1, 1) < max_depth
            small_depth_rot = z_rot.array.transpose(0, 2, 1).reshape(-1, 1) < max_depth
            warped = warped * small_depth
            warped_target = warped_target * small_depth
            warped_rot = warped_rot * small_depth_rot
            warped_rot_target = warped_rot_target * small_depth_rot

        if min_depth is not None:
            large_depth = z.array.transpose(0, 2, 1).reshape(-1, 1) > min_depth
            large_depth_rot = z_rot.array.transpose(0, 2, 1).reshape(-1, 1) > min_depth
            warped = warped * large_depth
            warped_target = warped_target * large_depth
            warped_rot = warped_rot * large_depth_rot
            warped_rot_target = warped_rot_target * large_depth_rot

        if self.norm == "l1":
            criteria = F.mean_absolute_error
        else:
            criteria = F.mean_squared_error
        loss = criteria(warped[:, :-1], warped_target[:, :-1]) + \
               criteria(warped_rot[:, :-1], warped_rot_target[:, :-1])
        loss += criteria(warped[:, -1], warped_target[:, -1]) * self.lambda_geometric + \
                criteria(warped_rot[:, -1], warped_rot_target[:, -1]) * self.lambda_geometric
        # print(loss.array)
        return loss, F.concat([new_zp, new_zp_rot], axis=0)

    def calc_real_pos(self, img, theta):
        xp = chainer.backends.cuda.get_array_module(img)
        if theta.ndim == 1:
            assert False, "only rotation matrices are supported for theta"
        else:
            R = theta[:, :3, :3]
            t = theta[:, :3, -1:]
        z = img[:, -1:].array.reshape(img.shape[0], 1, -1)
        rgb = img[:, :3].array.reshape(img.shape[0], 3, -1)
        real_pos = xp.matmul(xp.matmul(R, self.inv_K), z * self.p) + t
        return xp.concatenate([rgb, real_pos], axis=1)

    def occupancy_net_loss(self, occupancy_net, depth, theta, z):
        R = theta[:, :3, :3]
        t = theta[:, :3, -1:]
        depth = depth.reshape(depth.shape[0], 1, -1)
        eps = self.xp.random.normal(0, 0.05, size=depth.shape)
        real_pos = F.matmul(F.matmul(R, self.inv_K), (depth + eps) * self.p) + t
        label = (eps > 0).reshape(-1, 1).astype("int32")
        occupancy_field = occupancy_net(z, real_pos + eps)
        return F.sigmoid_cross_entropy(occupancy_field, label)


def warp(K, inv_K, R, t, z, p):
    # differentiable
    xp = chainer.cuda.get_array_module(K)
    new_zp = F.matmul(xp.matmul(xp.matmul(K, R), inv_K), z * p) - xp.matmul(xp.matmul(K, R), t)
    return new_zp.transpose(0, 2, 1)


def inv_warp(K, inv_K, inv_R, t, z, p):
    # differentiable
    xp = chainer.cuda.get_array_module(K)
    new_zp = F.matmul(xp.matmul(xp.matmul(K, inv_R), inv_K), z * p) + xp.matmul(K, t)
    return new_zp.transpose(0, 2, 1)


def bilinear(img, zp):
    # should be differentiable
    # https://arxiv.org/pdf/1904.04998.pdf のocclusion aware lossはRGBに対するlossからdepthに勾配が流れているため
    '''
    :param img: rgbd image b x 4 x h x w
    :param zp: depth * (x, y, 1)
    :return: interpolated colors
    '''
    assert isinstance(img, Variable), "img should be Variable"
    assert isinstance(zp, Variable), "img should be Variable"
    b, hw, _ = zp.shape
    _, _, h, w = img.shape
    xp = chainer.cuda.get_array_module(zp)
    zp = zp.reshape(-1, 3)  # (b x coords) x 3
    # neighborhood coordinates
    u = zp[:, 0] / F.clip(zp[:, 2], 1e-4, 10000)  # depth<0のところは無視
    v = zp[:, 1] / F.clip(zp[:, 2], 1e-4, 10000)

    v, u = u, v  # deepvoxelsではxとyの向きが逆
    u0 = u.array.astype("int32")
    u1 = u0 + 1
    v0 = v.array.astype("int32")
    v1 = v0 + 1

    # weights
    w1 = (u1 - u) * (v1 - v)
    w2 = (u - u0) * (v1 - v)
    w3 = (u1 - u) * (v - v0)
    w4 = (u - u0) * (v - v0)

    img_coord = xp.arange(b * hw) // hw
    not_getting_out = (u.array >= 0) * (u.array < h - 1) * \
                      (v.array >= 0) * (v.array < w - 1) * (zp[:, 2].array > 1e-4)  # はみ出さないpixel

    u0 = u0 * not_getting_out  # prevent illeagal access ()必要なのかわからんけど
    u1 = u0 * not_getting_out
    v0 = v0 * not_getting_out
    v1 = v1 * not_getting_out
    w1 = w1 * not_getting_out  # prevent illeagal access ()必要なのかわからんけど
    w2 = w2 * not_getting_out
    w3 = w3 * not_getting_out
    w4 = w4 * not_getting_out
    warped = (w1[:, None] * img[img_coord, :, u0, v0] + w2[:, None] * img[img_coord, :, u1, v0] +
              w3[:, None] * img[img_coord, :, u0, v1] + w4[:, None] * img[img_coord, :, u1, v1])
    return warped, not_getting_out  # warp先でのrgbd: (bhw x 4), はみ出したかどうか


def normal_map(zp, shape, inv_K):
    # shape: b x h x w
    xp = chainer.cuda.get_array_module(zp)
    abs_pos = xp.matmul(inv_K, zp)
    abs_pos = abs_pos.reshape(*shape, 3)
    # reflection paddingしてサイズを保って，convolutionする


class SmoothDepth:
    def __init__(self, xp):
        self.diff = xp.array([[[[0, 0, 0],
                                [1, -2, 1],
                                [0, 0, 0]]],
                              [[[0, 1, 0],
                                [0, -2, 0],
                                [0, 1, 0]]],
                              [[[0, 0, 0],
                                [1, -1, 0],
                                [-1, 1, 0]]]]).astype("float32")
        self.laplacian = xp.array([[[[1, 1, 1],
                                     [1, -8, 1],
                                     [1, 1, 1]]]]).astype("float32") / 8

    def __call__(self, x):
        x = F.average_pooling_2d(x, 2, 2, 0)
        depth_smoothness = F.convolution_2d(x, self.diff)
        depth_smoothness = F.sum(F.absolute(depth_smoothness), axis=1, keepdims=True)

        edge = F.convolution_2d(x, self.laplacian)
        loss = F.exp(-F.absolute(edge)) * depth_smoothness
        return F.mean(loss)
