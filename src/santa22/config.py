import math
import random

import numpy as np
from numba import njit
from tqdm import tqdm

from .utils import (
    get_origin,
    get_path_to_configuration,
    get_position,
    points_to_path,
    run_remove,
)


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


def standard_config_bottomright(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    assert x >= 0 and y < 0, "This function is only for the upper right quadrant"
    r = 64
    config = [(x - r, -r)]  # longest arm points to the right
    y = y - config[0][1]
    while r > 1:
        r = r // 2
        arm_y = np.clip(y, -r, r)
        config.append((r, arm_y))  # arm points upwards
        y -= arm_y
    arm_y = np.clip(y, -r, r)
    config.append((r, arm_y))  # arm points upwards
    assert y == arm_y
    return config


def standard_config_topleft(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    assert x <= 0 and y > 0, "This function is only for the upper left quadrant"
    # (_, 64), (-32, _), (-16, _) ,(-8, _), (-4, _), (-2, _), (-1, _), (-1, _)
    r = 64
    config = [(x - (-r), r)]  # longest arm points to the top
    y = y - config[0][1]
    while r > 1:
        r = r // 2
        arm_y = np.clip(y, -r, r)
        config.append((-r, arm_y))  # arm points leftwards
        y -= arm_y
    arm_y = np.clip(y, -r, r)
    config.append((-r, arm_y))  # arm points leftwards
    assert y == arm_y
    return config


def standard_config_bottomleft(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    assert x < 0 and y <= 0, "This function is only for the lower left quadrant"
    # (-64, _),(_, -32), (_, -16) ,(_, -8), (_, -4), (_, -2), (_, -1), (_, -1)
    r = 64
    config = [(-r, y - (-r))]  # longest arm points to the left
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, -r))  # arm points downwards
        x -= arm_x
    arm_x = np.clip(x, -r, r)
    config.append((arm_x, -r))  # arm points downwards
    assert x == arm_x
    return config


def standard_config(x, y):
    if x > 0 and y >= 0:
        return standard_config_topright(x, y)
    elif x >= 0 and y < 0:
        return standard_config_bottomright(x, y)
    elif x <= 0 and y > 0:
        return standard_config_topleft(x, y)
    elif x < 0 and y <= 0:
        return standard_config_bottomleft(x, y)
    else:
        return get_origin(257)


def get_baseline():
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

    return points_baseline


def improved_baseline():
    radius = 257 // 2

    cur_x = 0
    cur_y = 64
    # points = [(0, i) for i in range(65)]
    points = [(cur_x, cur_y)]

    for i in range((radius - cur_y) // 2):
        # go left
        for i in range(radius - 1):
            cur_x -= 1
            points.append((cur_x, cur_y))

        # go up
        cur_y += 1
        points.append((cur_x, cur_y))

        # go right
        for i in range(radius - 1):
            cur_x += 1
            points.append((cur_x, cur_y))

        # go up
        cur_y += 1
        points.append((cur_x, cur_y))

    # go to (-128, 128)
    for i in range(radius):
        cur_x -= 1
        points.append((cur_x, cur_y))

    # go to (-128, 65)
    for i in range(radius // 2 + 1):
        cur_y -= 1
        points.append((cur_x, cur_y))

    # go to (-128, -127)
    for i in range((radius - cur_y) // 2 * 3 - 1):
        # go right
        for i in range(radius - 1):
            cur_x += 1
            points.append((cur_x, cur_y))

        # go down
        cur_y -= 1
        points.append((cur_x, cur_y))

        # go left
        for i in range(radius - 1):
            cur_x -= 1
            points.append((cur_x, cur_y))

        # go up
        cur_y -= 1
        points.append((cur_x, cur_y))

    # snake to (-128, -1)
    for i in range(radius // 2 - 1):
        cur_y -= 1
        points.append((cur_x, cur_y))
        cur_x += 1
        points.append((cur_x, cur_y))
        cur_y += 1
        points.append((cur_x, cur_y))
        cur_x += 1
        points.append((cur_x, cur_y))

    cur_y -= 1
    points.append((cur_x, cur_y))
    cur_x += 1
    points.append((cur_x, cur_y))
    cur_y += 1
    points.append((cur_x, cur_y))
    cur_y -= 1
    points.append((cur_x, cur_y))

    # go to (-1, -128)
    cur_x += 1
    points.append((cur_x, cur_y))

    # go to (0, 0) (duplicated)
    for i in range((radius - cur_y) // 4):
        # go right
        for i in range(radius):
            cur_x += 1
            points.append((cur_x, cur_y))

        # go up
        cur_y += 1
        points.append((cur_x, cur_y))

        # go left
        for i in range(radius):
            cur_x -= 1
            points.append((cur_x, cur_y))

        # go up
        cur_y += 1
        points.append((cur_x, cur_y))

    # go to (1, 0)
    cur_x += 1
    points.append((cur_x, cur_y))
    # go to (2, 0)
    cur_x += 1
    points.append((cur_x, cur_y))

    # go to (1, 126)
    for i in range((radius - cur_y) // 2 - 1):
        # go right
        for i in range(radius - 2):
            cur_x += 1
            points.append((cur_x, cur_y))

        # go up
        cur_y += 1
        points.append((cur_x, cur_y))

        # go left
        for i in range(radius - 2):
            cur_x -= 1
            points.append((cur_x, cur_y))

        # go up
        cur_y += 1
        points.append((cur_x, cur_y))

    # go to (128, 127)
    # go right
    for i in range(radius - 2):
        cur_x += 1
        points.append((cur_x, cur_y))

    # go up
    cur_y += 1
    points.append((cur_x, cur_y))

    # snake to (2, 127)
    for i in range(radius // 2 - 1):
        cur_y += 1
        points.append((cur_x, cur_y))
        cur_x -= 1
        points.append((cur_x, cur_y))
        cur_y -= 1
        points.append((cur_x, cur_y))
        cur_x -= 1
        points.append((cur_x, cur_y))

    # snake to (1, 127)
    cur_y += 1
    points.append((cur_x, cur_y))
    cur_x -= 1
    points.append((cur_x, cur_y))

    # go to (1, 0)
    # go right
    for i in range(radius):
        cur_y -= 1
        points.append((cur_x, cur_y))

    return np.array(points)


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
def calc_threshold(improve, t_start, t_final, current_itr, max_itr):
    t = t_start + (t_final - t_start) * current_itr / max_itr
    return math.exp(improve / t)


@njit
def two_opt(points, offset, image_lut, t_start, t_end, itr, max_itr):
    i = random.randint(2, len(points) - (4 + offset))
    j = i + 1 + offset

    # A: i-1, B: i, C: j-1, D: j
    ac_idx = np.array([i - 1, j - 1])
    bd_idx = np.array([i, j])
    d_AB = evaluate_points(points[i - 1 : i + 1], image_lut)
    d_CD = evaluate_points(points[j - 1 : j + 1], image_lut)
    d_AC = evaluate_points(points[ac_idx], image_lut)
    d_BD = evaluate_points(points[bd_idx], image_lut)

    d0 = d_AB + d_CD
    d1 = d_AC + d_BD

    if d0 > d1 or random.random() < calc_threshold(
        d0 - d1, t_start, t_end, itr, max_itr
    ):
        return (
            np.concatenate(
                (
                    points[:i],  # ... A
                    points[i:j][::-1],  # CB
                    points[j:],  # D ...
                )
            ),
            -d0 + d1,
            d0 > d1,
        )

    return points, 0, False


def fill_gap_config(config):
    new_config = np.array(config[[0]])
    for i in tqdm(range(1, len(config))):
        if np.sum(np.abs(np.array(config[i - 1]) - np.array(config[i]))) != 1:
            tmp_config = get_path_to_configuration(config[i - 1], config[i])[1:]
            new_config = np.concatenate([new_config, tmp_config])
        else:
            new_config = np.concatenate([new_config, config[[i]]])
    return new_config


def local_search(image_lut, max_itr=10, t_start=0.3, t_end=0.001):

    start_configs = [
        [(64, i), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
        for i in range(64)
    ]
    start_points = [(0, i) for i in range(64)]

    initial_points = improved_baseline()
    initial_score = evaluate_points(initial_points, image_lut)
    print("initial color cost is ", initial_score)

    points = initial_points
    best_points = points
    current_score = initial_score
    best_score = initial_score
    tolerance_cnt = 0

    for itr in tqdm(range(max_itr)):

        offset = random.randint(0, 30)
        points_new, improve_score, improve_flag = two_opt(
            points, offset, image_lut, t_start, t_end, itr, max_itr
        )

        if improve_flag:
            tolerance_cnt = 0
        else:
            tolerance_cnt += 1

        if improve_score < 0:
            current_score = current_score + improve_score
            points = points_new

        if current_score < best_score:
            best_points = points.copy()
            best_score = current_score

    print(len(set([(p[0], p[1]) for p in best_points.tolist()] + start_points)))

    final_score = evaluate_points(best_points, image_lut)
    best_config = np.array(
        start_configs + [standard_config(p[0], p[1]) for p in best_points]
    )
    best_config = fill_gap_config(best_config)
    back_config = get_path_to_configuration(np.array(best_config[-1]), get_origin(257))[
        1:
    ]
    start_config = get_path_to_configuration(get_origin(257), np.array(best_config[0]))[
        :-1
    ]
    best_config = np.concatenate([start_config, best_config, back_config])
    best_config = run_remove(best_config)
    # best_config = run_remove(best_config)

    print("improved score is ", final_score)
    return best_config, best_points, initial_score > final_score
