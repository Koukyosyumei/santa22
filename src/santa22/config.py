import random

import numpy as np
from numba import njit


@njit
def standard_config_topright(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)
    [(64, _), (_, 32), (_, 16), (_, 8), (_, 4), (_, 2), (_, 1), (_, 1)]
    """
    assert x > 0 and y >= 0, "This function is only for the upper right quadrant"
    r = 64
    config = [(r, y - r)]  # longest arm points to the right
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, r))  # arm points upwards
        x -= arm_x
    arm_x = np.clip(x, -r, r)
    config.append((arm_x, r))  # arm points upwards
    assert x == arm_x
    return config


@njit
def standard_config_botoomright(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)
    [(64, _), (_, -32), (_, -16), (_, -8), (_, -4), (_, -2), (_, -1), (_, -1)]
    """
    assert x > 0 and y < 0, "This function is only for the lower right quadrant"
    r = 64
    config = [(r, y + r)]  # longest arm points to the right
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, -r))  # arm points bottom
        x -= arm_x
    arm_x = np.clip(x, -r, r)
    config.append((arm_x, -r))  # arm points bottom
    assert x == arm_x
    return config


@njit
def standard_config_topleft(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)
    [(-64, _), (_, 32), (_, 16), (_, 8), (_, 4), (_, 2), (_, 1), (_, 1)]
    """
    assert x <= 0 and y >= 0, "This function is only for the upper left quadrant"
    r = 64
    config = [(-r, y - r)]  # longest arm points to the right
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, r))  # arm points upwards
        x -= arm_x
    arm_x = np.clip(x, -r, r)
    config.append((arm_x, r))  # arm points upwards
    assert x == arm_x
    return config


@njit
def standard_config_botoomleft(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)
    [(-64, _), (_, -32), (_, -16), (_, -8), (_, -4), (_, -2), (_, -1), (_, -1)]
    """
    assert x <= 0 and y < 0, "This function is only for the lower right quadrant"
    r = 64
    config = [(-r, y + r)]  # longest arm points to the right
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, -r))  # arm points bottom
        x -= arm_x
    arm_x = np.clip(x, -r, r)
    config.append((arm_x, -r))  # arm points bottom
    assert x == arm_x
    return config


def get_baseline():
    # Generate points
    points_baseline = []
    flag = True
    for split in range(2):
        for i in reversed(range(257)) if split % 2 == 0 else range(257):
            if not flag:
                for j in range(128 * split, 128 + 129 * split):
                    points_baseline.append((j - 128, i - 128))
            else:
                for j in reversed(range(128 * split, 128 + 129 * split)):
                    points_baseline.append((j - 128, i - 128))
            flag = not flag
        flag = False
    points_baseline = np.array(points_baseline)


@njit
def pos2lut_idx(pos):
    """Convert positions in the range of [-128, 128] into row index for the RGB-LUT"""
    transformed_pos = pos + 128
    return transformed_pos[:, 0] + (256 - transformed_pos[:, 1]) * 257


@njit
def evaluate_points(points, image_lut):
    """Generates the RGB-path from the configuration matrix and calls the cost function"""
    lut_idx = pos2lut_idx(points)
    rgb_path = image_lut[lut_idx, -3:]
    return np.abs(rgb_path[:-1, :] - rgb_path[1:, :]).sum()


@njit
def two_opt(points, offset, image_lut, t_start, t_end, itr, max_itr):
    i = random.randint(1, len(points) - (3 + offset))
    j = i + 1 + offset

    d_AB = evaluate_points(points[i - 1 : i + 1], image_lut)
    d_CD = evaluate_points(points[j - 1 : j + 1], image_lut)
    d_AC = evaluate_points(points[[i - 1, j - 1]], image_lut)
    d_BD = evaluate_points(points[[i, j]], image_lut)

    d0 = d_AB + d_CD
    d1 = d_AC + d_BD

    if d0 > d1 or random.random() < calc_threshold(
        d0 - d1, t_start, t_end, itr, max_itr
    ):
        return (
            np.concatenate(
                (
                    points[: i - 1],
                    points[[i - 1, j - 1]],
                    points[i:j][::-1][1:],
                    points[j:],
                )
            ),
            -d0 + d1,
            d0 > d1,
        )

    return points, 0, False
