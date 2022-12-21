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


def config_to_string(config):
    return ";".join([" ".join(map(str, vector)) for vector in config])
