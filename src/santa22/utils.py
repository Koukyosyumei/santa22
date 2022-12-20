import io
import os
import pickle
from itertools import product
from pathlib import Path
from numba import jit
import cv2
import numpy as np
import pandas as pd

# Functions to map between cartesian coordinates and array indexes


@jit
def cartesian_to_array(x, y, shape_):
    m, n = shape_[:2]
    i_ = (n - 1) // 2 - y
    j = (n - 1) // 2 + x
    if i_ < 0 or i_ >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    return i_, j


@jit
def array_to_cartesian(i_, j, shape_):
    m, n = shape_[:2]
    if i_ < 0 or i_ >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    y = (n - 1) // 2 - i_
    x = j - (n - 1) // 2
    return x, y


# Functions to map an image between array and record formats


def image_to_dict(image_):
    image_ = np.atleast_3d(image_)
    kv_image = {}
    for i_, j in product(range(len(image_)), repeat=2):
        kv_image[array_to_cartesian(
            i_, j, image_.shape)] = tuple(image_[i_, j])
    return kv_image


def image_to_df(image_):
    return pd.DataFrame(
        [(x, y, r, g, b)
         for (x, y), (r, g, b) in image_to_dict(image_).items()],
        columns=["x", "y", "r", "g", "b"],
    )


def df_to_image(df):
    side = int(len(df) ** 0.5)  # assumes a square image
    return df.set_index(["x", "y"]).to_numpy().reshape(side, side, -1)


def imread(path):
    if isinstance(path, Path):
        path = path.as_posix()
    return cv2.imread(path)[:, :, ::-1] / 255


def check_point(output_dir, current_solutions_cost_, current_solution_):
    """Makes check-points during the search"""
    with io.open(
        os.path.join(output_dir, "current_solution_data.pkl"), "wb"
    ) as out_file:
        pickle.dump(
            {"cost": current_solutions_cost_,
                "solution": current_solution_}, out_file
        )


def config_to_string(config):
    """Converts the path generated from the new
    best solution into the submission format
    """
    return ";".join([" ".join(map(str, vector)) for vector in config])
