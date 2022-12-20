import numpy as np


def get_position(config):
    return config.sum(0)


def rotate_link(vector, direction):
    x, y = vector
    if direction == 1:  # counter-clockwise
        if y >= x and y > -x:
            x -= 1
        elif x < y <= -x:
            y -= 1
        elif y <= x and y < -x:
            x += 1
        else:
            y += 1
    elif direction == -1:  # clockwise
        if y > x and y >= -x:
            x += 1
        elif x <= y < -x:
            y += 1
        elif y < x and y <= -x:
            x -= 1
        else:
            y -= 1
    return x, y


def rotate(config, i_, direction):
    config = config.copy()
    config[i_] = rotate_link(config[i_], direction)
    return config


def compress_path(path):
    """
    Compress a path between two points
    """
    n_joints = path.shape[1]
    r = np.zeros((n_joints, path.shape[0], 2), dtype=path.dtype)
    l = np.zeros(n_joints, dtype="int")
    for j in range(len(path)):
        for i_ in range(n_joints):
            if l[i_] == 0 or (r[i_][l[i_] - 1] != path[j, i_]).any():
                r[i_, l[i_]] = path[j, i_]
                l[i_] += 1
    r = r[:, : l.max()]

    for i_ in range(n_joints):
        for j in range(l[i_], r.shape[1]):
            r[i_, j] = r[i_, j - 1]
    r = r.transpose(1, 0, 2)

    return r


def get_direction(u, v):
    """Returns the sign of the angle from u to v."""
    # direction = np.sign(np.cross(u, v))
    direction = np.sign(u[0] * v[1] - u[1] * v[0])
    if direction == 0 and (u * v).sum() < 0:
        direction = 1
    return direction


def get_radius(config):
    r = 0
    for link in config:
        r += np.abs(link).max()
    return r


def get_radii(config):
    radii = np.cumsum(np.maximum(np.abs(config[:, 0]), np.abs(config[:, 1]))[::-1])[
        ::-1
    ]
    return np.append(radii, np.zeros(1, dtype="int"))
