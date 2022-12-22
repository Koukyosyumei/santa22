import os
from functools import reduce

import numpy as np
from numba import njit


@njit
def get_position(config):
    return config.sum(
        0
    )  # reduce(lambda p, q: (p[0] + q[0], p[1] + q[1]), config, (0, 0))


@njit
def compress_path(path):
    """
    Compress a path between two points
    """
    n_joints = path.shape[1]
    r = np.zeros((n_joints, path.shape[0], 2), dtype=path.dtype)
    ll = np.zeros(n_joints, dtype="int")
    for j in range(len(path)):
        for i_ in range(n_joints):
            if ll[i_] == 0 or (r[i_][ll[i_] - 1] != path[j, i_]).any():
                r[i_, ll[i_]] = path[j, i_]
                ll[i_] += 1
    r = r[:, : ll.max()]

    for i_ in range(n_joints):
        for j in range(ll[i_], r.shape[1]):
            r[i_, j] = r[i_, j - 1]
    r = r.transpose(1, 0, 2)

    return r


@njit
def cartesian_to_array(x, y, shape_):
    m, n = shape_[:2]
    i_ = (n - 1) // 2 - y
    j = (n - 1) // 2 + x
    if i_ < 0 or i_ >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    return i_, j


@njit
def rotate_link(vector, direction):
    x, y = vector
    if direction == 1:  # counter-clockwise
        if y >= x and y > -x:
            x -= 1
        elif y > x and y <= -x:
            y -= 1
        elif y <= x and y < -x:
            x += 1
        else:
            y += 1
    elif direction == -1:  # clockwise
        if y > x and y >= -x:
            x += 1
        elif y >= x and y < -x:
            y += 1
        elif y < x and y <= -x:
            x -= 1
        else:
            y -= 1
    return x, y


@njit
def rotate(config, i, direction):
    config = config.copy()
    config[i] = rotate_link(config[i], direction)
    return config


@njit
def get_direction(u, v):
    """Returns the sign of the angle from u to v."""
    # direction = np.sign(np.cross(u, v))
    direction = np.sign(u[0] * v[1] - u[1] * v[0])
    if direction == 0 and (u * v).sum() < 0:
        direction = 1
    return direction


@njit
def get_radius(config):
    r = 0
    for link in config:
        r += np.abs(link).max()
    return r


@njit
def get_radii(config):
    radii = np.cumsum(np.maximum(np.abs(config[:, 0]), np.abs(config[:, 1]))[::-1])[
        ::-1
    ]
    return np.append(radii, np.zeros(1, dtype="int"))


@njit
def get_path_to_point(config, point_):
    """Find a path of configurations to `point` starting at `config`."""
    config_start = config.copy()
    radii = get_radii(config)

    # Rotate each link, starting with the largest, until the point can
    # be reached by the remaining links. The last link must reach the
    # point itself.
    for i_ in range(len(config)):
        link = config[i_]
        base = get_position(config[:i_])
        relbase = point_ - base
        position = get_position(config[: i_ + 1])
        relpos = point_ - position
        radius = radii[i_ + 1]

        # Special case when next-to-last link lands on point.
        if radius == 1 and (relpos == 0).all():
            config = rotate(config, i_, 1)
            if (get_position(config) == point_).all():
                break
            else:
                continue
        while np.max(np.abs(relpos)) > radius:
            direction = get_direction(link, relbase)
            config = rotate(config, i_, direction)
            link = config[i_]
            base = get_position(config[:i_])
            relbase = point_ - base
            position = get_position(config[: i_ + 1])
            relpos = point_ - position
            radius = get_radius(config[i_ + 1 :])

    assert (get_position(config) == point_).all()
    path = get_path_to_configuration(config_start, config)
    path = compress_path(path)

    return path


@njit
def get_path_to_configuration(from_config, to_config):
    path = np.expand_dims(from_config, 0).copy()
    config = from_config.copy()
    while (config != to_config).any():
        for i_ in range(len(config)):
            config = rotate(config, i_, get_direction(config[i_], to_config[i_]))
        path = np.append(path, np.expand_dims(config, 0), 0)
    assert (path[-1] == to_config).all()
    path = compress_path(path)
    return path


def get_origin(size):
    assert size % 2 == 1
    radius = size // 2
    p = [1]
    for power in range(0, 8):
        p.append(2**power)
        if sum(p) == radius:
            break
    else:
        assert False
    p = p[::-1]
    config = np.array([(p[0], 0)] + [(-pp, 0) for pp in p[1:]])
    return config


def points_to_path(points, size=257):
    origin = get_origin(size)
    visited = set()
    path = [origin]
    for p in points:
        config = path[-1]
        if tuple(p) not in visited:
            candy_cane_road = get_path_to_point(np.array(config), np.array(p))[1:]
            if len(candy_cane_road) > 0:
                visited |= set(
                    [tuple(get_position(np.array(r))) for r in candy_cane_road]
                )
            path.extend(candy_cane_road)
    # Back to origin
    candy_cane_road = get_path_to_configuration(np.array(path[-1]), origin)[1:]
    visited |= set([tuple(get_position(np.array(r))) for r in candy_cane_road])
    path.extend(candy_cane_road)

    assert (
        len(visited) == size**2
    ), f"Visited {len(visited)} points out of {size ** 2}"

    return np.array(path)


def find_duplicate_points(path):

    duplicate_points = {}
    for c in path:
        p = tuple(get_position(c))
        if p != (0, 0):
            duplicate_points[p] = duplicate_points.get(p, 0) + 1

    return duplicate_points


def vector_diff_one(path):

    for i in range(len(path) - 1):
        for c0, c1 in zip(path[i], path[i + 1]):
            if abs(c0[0] - c1[0]) + abs(c0[1] - c1[1]) > 1:
                return False

    return True


def run_remove(path):
    duplicate_points = find_duplicate_points(path)

    i = len(path) - 2
    while i >= 0:
        local_p = path[i : i + 3]
        p = tuple(get_position(local_p[1]))
        new_local_p = compress_path(local_p)
        if (
            vector_diff_one(new_local_p)
            and duplicate_points.get(p, 0) > 1
            and len(new_local_p) < 3
        ):
            path = np.concatenate((path[: i + 1], path[i + 2 :]))
            duplicate_points[p] -= 1
        i -= 1

    return path


def config_to_string(config):
    return ";".join([" ".join(map(str, vector)) for vector in config])


def save_config(output_dir, file_name, config):
    f = open(os.path.join(output_dir, file_name), "w")
    print("configuration", file=f)  # header
    for c in config:
        print(config_to_string(c), file=f)
