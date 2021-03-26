import math

gpu_lr = {
    1: {15: 1.5, 16: 1.5, 17: 1.5},
    2: {13: 1.5, 14: 1.5, 15: 2, 16: 2, 17: 2},
    3: {11: 1.5, 12: 1.5, 13: 2, 14: 2, 15: 2.5, 16: 2.5, 17: 2.5},
    4: {11: 1.5, 12: 1.5, 13: 2, 14: 2, 15: 3, 16: 3, 17: 3},
    8: {9: 1.5, 10: 1.5, 11: 2, 12: 2, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3},
}


def get_lr_scale_factor(total_gpu, stage):
    gpu_lr_d = gpu_lr.get(total_gpu, gpu_lr[1])
    stage = math.floor(stage)
    if stage >= 18:
        return gpu_lr_d[17]
    return gpu_lr_d.get(stage, 1)
