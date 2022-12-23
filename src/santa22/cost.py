import numpy as np
from numba import njit

from .utils import cartesian_to_array, get_position


@njit
def reconfiguration_cost(from_config, to_config):
    diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
    assert diffs.max() <= 1
    return float(np.sqrt(diffs.sum()))


@njit
def color_cost(from_position, to_position, image, color_scale=3.0):
    return np.abs(image[to_position] - image[from_position]).sum() * color_scale


@njit
def color_near_cost(from_position, to_position, image, color_scale=0.003, offset=1):
    return (
        np.abs(
            image[
                max(0, to_position[0] - offset) : min(
                    image.shape[0], to_position[0] + 1 + offset
                ),
                max(0, to_position[1] - offset) : min(
                    image.shape[1], to_position[1] + 1 + offset
                ),
            ]
            - image[from_position]
        ).sum()
        * color_scale
    )


@njit
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


@njit
def total_cost(path, image_):
    """
    Computes total cost of path over image
    """
    cost = 0.0
    for i_ in range(1, len(path)):
        cost += step_cost(path[i_ - 1], path[i_], image_)
    return cost


@njit
def pos2lut_idx(pos):
    """Convert positions in the range of [-128, 128] into row index for the RGB-LUT"""
    transformed_pos = pos + 128
    return transformed_pos[:, 0] + (256 - transformed_pos[:, 1]) * 257


@njit
def cost_fun(config, rgb_path):
    """This cost function takes the configuration matrix and the corresponding visited
    colors of the path as input and returns the scalar float cost"""
    return np.sqrt(
        np.abs(config[:-1, :, :] - config[1:, :, :]).sum(axis=-1).sum(axis=-1)
    ).sum() + (3.0 * np.abs(rgb_path[:-1, :] - rgb_path[1:, :]).sum())


@njit
def evaluate_config(config, image_lut):
    """Generates the RGB-path from the configuration matrix and calls the cost function"""
    lut_idx = pos2lut_idx(config.sum(axis=1))
    rgb_path = image_lut[lut_idx, -3:]
    return cost_fun(config, rgb_path)
