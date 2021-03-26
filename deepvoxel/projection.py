import numpy as np
from chainer.backends import cuda
from chainer import functions as F


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

    # def depth_to_skeleton(self, ux, uy, depth):
    #     '''Given a point in pixel coordinates plus depth gives the coordinates of the imaged point in camera coordinates
    #     '''
    #     xp = cuda.get_array_module(ux)
    #     x = (ux - self.lifting_intrinsic[0][2]) / self.lifting_intrinsic[0][0]
    #     y = (uy - self.lifting_intrinsic[1][2]) / self.lifting_intrinsic[1][1]
    #     return xp.array([depth * x, depth * y, depth])

    def skeleton_to_depth(self, p):
        '''Given a point in camera coordinates gives the pixel coordinates of the projected point plus depth
        '''
        x = (p[0] * self.lifting_intrinsic[0][0]) / p[2] + self.lifting_intrinsic[0][2]
        y = (p[1] * self.lifting_intrinsic[1][1]) / p[2] + self.lifting_intrinsic[1][2]
        return F.concat([x, y, p[2]], axis=0)

    # def compute_frustum_bounds(self, camera_to_world, world_to_grid):
    #     # calculate corner points in camera coordinates
    #     corner_points = camera_to_world.new(8, 4, 1).fill_(1)
    #
    #     # depth min
    #     corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min).unsqueeze(1)
    #     corner_points[1][:3] = self.depth_to_skeleton(self.lifting_image_dims[0] - 1, 0, self.depth_min).unsqueeze(1)
    #     corner_points[2][:3] = self.depth_to_skeleton(self.lifting_image_dims[0] - 1, self.lifting_image_dims[1] - 1,
    #                                                   self.depth_min).unsqueeze(1)
    #     corner_points[3][:3] = self.depth_to_skeleton(0, self.lifting_image_dims[1] - 1, self.depth_min).unsqueeze(1)
    #     # depth max
    #     corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max).unsqueeze(1)
    #     corner_points[5][:3] = self.depth_to_skeleton(self.lifting_image_dims[0] - 1, 0, self.depth_max).unsqueeze(1)
    #     corner_points[6][:3] = self.depth_to_skeleton(self.lifting_image_dims[0] - 1, self.lifting_image_dims[1] - 1,
    #                                                   self.depth_max).unsqueeze(1)
    #     corner_points[7][:3] = self.depth_to_skeleton(0, self.lifting_image_dims[1] - 1, self.depth_max).unsqueeze(1)
    #
    #     # Transform to world coordinates
    #     p = torch.bmm(camera_to_world.repeat(8, 1, 1), corner_points)
    #     # Transform to grid coordinates (grid at origin)
    #     pl = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.floor(p)))
    #     pu = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.ceil(p)))
    #     bbox_min0, _ = torch.min(pl[:, :3, 0], 0)
    #     bbox_min1, _ = torch.min(pu[:, :3, 0], 0)
    #     bbox_min = torch.min(bbox_min0, bbox_min1)
    #     bbox_max0, _ = torch.max(pl[:, :3, 0], 0)
    #     bbox_max1, _ = torch.max(pu[:, :3, 0], 0)
    #     bbox_max = torch.max(bbox_max0, bbox_max1)
    #     return bbox_min, bbox_max, pl

    # def comp_lifting_idcs(self, camera_to_world, grid2world):
    #     xp = cuda.get_array_module(camera_to_world)
    #     world2cam = xp.linalg.inv(camera_to_world)  # inverse matrix
    #     world2grid = xp.linalg.inv(grid2world)
    #
    #     # Voxel bounds are computed in the grid coordinate system (grid at origin)
    #     voxel_bounds_min, voxel_bounds_max, _ = self.compute_frustum_bounds(camera_to_world, world2grid)
    #     voxel_bounds_min = voxel_bounds_min.to(self.device)
    #     voxel_bounds_max = voxel_bounds_max.to(self.device)
    #
    #     # Linear indices into the volume
    #     lin_ind_volume = torch.arange(0, self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2],
    #                                   out=torch.LongTensor()).to(self.device)
    #
    #     # 4-vector for each image feature (batch_size*number_of_views)
    #     coords = camera_to_world.new(4, lin_ind_volume.size(0)).to(self.device)
    #
    #     # Manually compute x-y-z voxel coordinates of volume
    #     coords[2] = lin_ind_volume / (self.grid_dims[0] * self.grid_dims[1])
    #     tmp = lin_ind_volume - (coords[2] * self.grid_dims[0] * self.grid_dims[1]).long().to(self.device)
    #     coords[1] = tmp / self.grid_dims[0]
    #     coords[0] = torch.remainder(tmp, self.grid_dims[0])
    #     coords[3].fill_(1)
    #
    #     # Volume is centered around origin
    #     coords[0] -= self.grid_dims[0] / 2
    #     coords[1] -= self.grid_dims[1] / 2
    #     coords[2] -= self.grid_dims[2] / 2
    #
    #     # Transform voxel coordinates into meters
    #     coords[:3, :] *= self.voxel_size
    #
    #     # Everything that's outside the frustrum gets the boot
    #     mask_frustum_bounds = (torch.ge(coords[0], voxel_bounds_min[0]) *
    #                            torch.ge(coords[1], voxel_bounds_min[1]) *
    #                            torch.ge(coords[2], voxel_bounds_min[2]))
    #     mask_frustum_bounds = (mask_frustum_bounds *
    #                            torch.lt(coords[0], voxel_bounds_max[0]) *
    #                            torch.lt(coords[1], voxel_bounds_max[1]) *
    #                            torch.lt(coords[2], voxel_bounds_max[2]))
    #
    #     if not mask_frustum_bounds.any():
    #         print('error: nothing in frustum bounds')
    #         return None
    #
    #     lin_ind_volume = lin_ind_volume[mask_frustum_bounds]
    #
    #     # Recompute the coordinate array with the fewer, valid indices
    #     coords = coords.resize_(4, lin_ind_volume.size(0))
    #     coords[2] = lin_ind_volume / (self.grid_dims[0] * self.grid_dims[1])
    #     tmp = lin_ind_volume - (coords[2] * self.grid_dims[0] * self.grid_dims[1]).long().to(self.device)
    #     coords[1] = tmp / self.grid_dims[0]
    #     coords[0] = torch.remainder(tmp, self.grid_dims[0])
    #     coords[3].fill_(1)
    #
    #     coords[0] -= self.grid_dims[0] // 2
    #     coords[1] -= self.grid_dims[1] // 2
    #     coords[2] -= self.grid_dims[2] // 2
    #
    #     # Transform voxel coordinates into meters
    #     coords[:3, :] *= self.voxel_size
    #
    #     # transform grid coordinates to current frame
    #     p = torch.mm(world2cam, torch.mm(grid2world, coords.float())).to(self.device)
    #
    #     # project to pixel coordinates
    #     p[0] = (p[0] * self.lifting_intrinsic[0][0]) / p[2] + self.lifting_intrinsic[0][2]
    #     p[1] = (p[1] * self.lifting_intrinsic[1][1]) / p[2] + self.lifting_intrinsic[1][2]
    #     pi = p.round().long()
    #
    #     # Everything that's out of the image boundaries gets the boot # TODO
    #     valid_ind_mask = (torch.ge(pi[0], 0) *
    #                       torch.ge(pi[1], 0) *
    #                       torch.lt(pi[0], self.lifting_image_dims[0]) *
    #                       torch.lt(pi[1], self.lifting_image_dims[1]))
    #     if not valid_ind_mask.any():
    #         print('error: no valid image indices')
    #         return None
    #
    #     # Update p and the volume indices
    #     valid_p = p[:, valid_ind_mask]
    #     lin_ind_volume = lin_ind_volume[valid_ind_mask]
    #
    #     final_lin_ind = lin_ind_volume
    #     interpolation_coordinates = valid_p[:3, :]
    #
    #     return final_lin_ind, interpolation_coordinates

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

# def interpolate_lifting(image, lin_ind_3d, query_points, grid_dims, xp):
#     batch, num_feats, height, width = image.shape
#
#     image = image.cuda()
#     lin_ind_3d = xp.array(lin_ind_3d)
#     query_points = xp.array(query_points)
#
#     x_indices = query_points[1, :]
#     y_indices = query_points[0, :]
#
#     x0 = x_indices.astype("int32")
#     y0 = y_indices.astype("int32")
#
#     x1 = (x0 + 1)
#     y1 = (y0 + 1)
#
#     x1 = xp.clip(x1, 0, width - 1)
#     y1 = xp.clip(y1, 0, height - 1)
#
#     x = x_indices - x0
#     y = y_indices - y0
#
#     output = xp.zeros(1, num_feats, grid_dims[0] * grid_dims[1] * grid_dims[2]).cuda()
#     output[:, :, lin_ind_3d] += image[:, :, x0, y0] * (1 - x) * (1 - y)
#     output[:, :, lin_ind_3d] += image[:, :, x1, y0] * x * (1 - y)
#     output[:, :, lin_ind_3d] += image[:, :, x0, y1] * (1 - x) * y
#     output[:, :, lin_ind_3d] += image[:, :, x1, y1] * x * y
#
#     output = output.view(batch, num_feats, grid_dims[0], grid_dims[1], grid_dims[2])  # Width first
#
#     return output
