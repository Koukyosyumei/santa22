import numpy as np
from numba import jit

from .utils import cartesian_to_array, get_position


@jit
def reconfiguration_cost(from_config, to_config):
    diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
    assert diffs.max() <= 1
    return float(np.sqrt(diffs.sum()))


@jit
def color_cost(from_position, to_position, image, color_scale=3.0):
    return np.abs(image[to_position] - image[from_position]).sum() * color_scale


@jit
def step_cost(from_config, to_config, image_):
    """
    Total cost of one step: the reconfiguration cost plus the color cost
    """
    pos_from = get_position(from_config)
    pos_to = get_position(to_config)
    from_position = cartesian_to_array(pos_from[0], pos_from[1], image_.shape)
    to_position = cartesian_to_array(pos_to[0], pos_to[1], image_.shape)
    return reconfiguration_cost(from_config, to_config) + color_cost(
        from_position, to_position, image_
    )


@jit
def total_cost(path, image_):
    """
    Computes total cost of path over image
    """
    cost = 0.0
    for i_ in range(1, len(path)):
        cost += step_cost(path[i_ - 1], path[i_], image_)
    return cost
