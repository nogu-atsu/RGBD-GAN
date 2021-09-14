from chainer import functions as F
from chainer.backends import cuda


class ProjectionHelper:
    def __init__(self,
                 lifting_intrinsic,
                 projection_intrinsic,
                 projection_image_dims,
                 lifting_image_dims,
                 depth_min,
                 depth_max,
                 grid_dims,
                 voxel_size,
                 near_plane,
                 frustrum_depth,
                 device):
        self.grid_dims = grid_dims
        self.projection_intrinsic = projection_intrinsic
        self.lifting_intrinsic = lifting_intrinsic

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.projection_image_dims = projection_image_dims
        self.lifting_image_dims = lifting_image_dims
        self.voxel_size = voxel_size
        self.device = device
        self.near_plane = near_plane
        self.frustrum_depth = frustrum_depth

        print("\n" + "*" * 100)
        print("Lifting intrinsic is %s" % self.lifting_intrinsic)
        print("Projection intrinsic is %s" % self.projection_intrinsic)

        print("Lifting image dims is ", self.lifting_image_dims)
        print("Projection image dims is ", self.projection_image_dims)

        print("voxel size is %s" % self.voxel_size)
        print("*" * 100 + "\n")

    def skeleton_to_depth(self, p):
        '''Given a point in camera coordinates gives the pixel coordinates of the projected point plus depth
        '''
        x = (p[0] * self.lifting_intrinsic[0][0]) / p[2] + self.lifting_intrinsic[0][2]
        y = (p[1] * self.lifting_intrinsic[1][1]) / p[2] + self.lifting_intrinsic[1][2]
        return F.concat([x, y, p[2]], axis=0)

    def compute_proj_idcs(self, cam2world, grid2world=None):
        # Linear index into the frustrum
        # lin_ind_frustrum = torch.arange(0, self.image_dims[0]*self.image_dims[1]*self.grid_dims[2]).long().cuda()
        # indexを指定するだけなので，微分可能である必要はない
        xp = cuda.get_array_module(cam2world)
        if grid2world is not None:
            world2grid = xp.linalg.inv(grid2world)

        num_frust_elements = self.projection_image_dims[0] * self.projection_image_dims[1] * int(self.frustrum_depth)

        lin_ind_frustrum = xp.arange(0, num_frust_elements).astype("int32")

        coords = xp.zeros((4, num_frust_elements), dtype="float32")

        # Manually compute x-y-z voxel coordinates of volume
        # 画像サイズ x 画像サイズ x frustrum_depthのindexを計算しているだけ
        coords[2] = lin_ind_frustrum // (self.projection_image_dims[0] * self.projection_image_dims[1])
        tmp = lin_ind_frustrum - \
              (coords[2] * self.projection_image_dims[0] * self.projection_image_dims[1]).astype("int32")
        coords[1] = tmp / self.projection_image_dims[0]
        coords[0] = tmp % self.projection_image_dims[0]
        coords[3].fill(1)

        # Map the z-coordinates to different z-planes
        # frustrumのz座標を計算している
        coords[2] *= self.voxel_size
        coords[2] += self.near_plane

        # 画像座標系からcamera座標系のx,y座標への変換
        # K^(-1)zpの計算
        coords[0] = (coords[0] - self.projection_intrinsic[0][2]) / self.projection_intrinsic[0][0]
        coords[1] = (coords[1] - self.projection_intrinsic[1][2]) / self.projection_intrinsic[1][1]
        coords[:2] *= coords[2]
        # cameraの回転行列をかけて,world座標系に変換
        grid_coords = xp.dot(cam2world, coords)
        if grid2world is not None:
            grid_coords = xp.dot(world2grid, grid_coords)

        # world座標をvoxelのindexに変換
        voxel_coords = grid_coords[:3, :] / self.voxel_size
        voxel_coords = (voxel_coords + self.grid_dims[2] / 2)

        # Everything that's outside the frustrum gets the boot
        # 各voxelではみ出しているところを計算
        mask_frustrum_bounds = xp.all(voxel_coords >= 0, axis=0)
        mask_frustrum_bounds = (mask_frustrum_bounds *
                                (voxel_coords[0] < self.grid_dims[0]) *
                                (voxel_coords[1] < self.grid_dims[1]) *
                                (voxel_coords[2] < self.grid_dims[2]))

        if not mask_frustrum_bounds.any():
            print('error: nothing in frustum bounds')
            return None

        lin_ind_frustrum = lin_ind_frustrum[mask_frustrum_bounds]
        voxel_coords = voxel_coords[:, mask_frustrum_bounds]

        return lin_ind_frustrum, voxel_coords
