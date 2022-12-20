import numpy as np


def pos2lut_idx(pos):
    """Convert positions in the range of [-128, 128]
    into row index for the RGB-LUT"""
    transformed_pos = pos + 128
    return transformed_pos[:, 0] + (256 - transformed_pos[:, 1]) * 257


def cost_fun(config, rgb_path):
    """This cost function takes the configuration matrix
    and the corresponding visited
    colors of the path as input and returns
    the scalar float cost"""
    return np.sqrt(
        np.abs(config[:-1, :, :] - config[1:, :, :]).sum(axis=-1).sum(axis=-1)
    ).sum() + (3.0 * np.abs(rgb_path[:-1, :] - rgb_path[1:, :]).sum())


def evaluate_config(config, image_lut):
    """Generates the RGB-path from the configuration
    matrix and calls the cost function"""
    lut_idx = pos2lut_idx(config.sum(axis=1))
    rgb_path = image_lut[lut_idx, -3:]
    return cost_fun(config, rgb_path)
