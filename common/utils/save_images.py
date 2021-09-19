import numpy as np

try:
    import cupy
except:
    pass


def convert_batch_images(x, rows, cols):
    rgbd = False
    if x.shape[1] == 4:
        rgbd = True
        depth = np.tile(x[:, -1:], (1, 3, 1, 1))
        x = x[:, :-1]
    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    if rgbd:
        depth = np.asarray(np.clip(1 / (depth) * 128, 0.0, 255.0), dtype=np.uint8)
        depth = depth.reshape((rows, cols, 3, H, W))
        x = np.concatenate([x, depth], axis=1).reshape(rows * 2, cols, 3, H, W)
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((-1, cols * W, 3))
    return x
