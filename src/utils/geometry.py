import math

import numpy as np


def dist_3d(p1, p2):
    return float(math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    ))


def interpolate_segment_3d(p1, p2, resolution=5.0):
    length = dist_3d(p1, p2)
    n = max(2, int(length / resolution) + 1)
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    zs = np.linspace(p1[2], p2[2], n)
    return list(zip(xs, ys, zs))


def path_length_3d(path_xyz):
    if path_xyz is None or len(path_xyz) < 2:
        return 0.0
    return float(sum(dist_3d(path_xyz[i], path_xyz[i + 1]) for i in range(len(path_xyz) - 1)))