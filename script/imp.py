import argparse
import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm, trange

from santa22.config import local_search, plot_traj, standard_config
from santa22.cost import evaluate_config
from santa22.utils import get_origin, get_path_to_configuration, save_config


def imread(path):
    if isinstance(path, Path):
        path = path.as_posix()
    return cv2.imread(path)[:, :, ::-1] / 255


def df_to_imagelut(image_df):
    return (image_df + np.array([[128, 128, 0, 0, 0]])).to_numpy()


def add_args(parser):
    parser.add_argument(
        "-d",
        "--data_dir",
        default="/kaggle/input/santa-2022",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=".",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--initial_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-e",
        "--epsilon_greedy",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "-r",
        "--epsilon_local_search",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "-i",
        "--max_itr",
        default=100000,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--t_start",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "--t_end",
        default=0.001,
        type=float,
    )
    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    print(f"Solve Santa2022 with greedy & local search (seed={parsed_args.seed})")

    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)

    df = pd.read_csv(os.path.join(parsed_args.data_dir, "image.csv"))
    image_lut = df_to_imagelut(df)

    final_config, final_points, _ = local_search(
        image_lut, parsed_args.max_itr, parsed_args.t_start, parsed_args.t_end
    )

    print(evaluate_config(final_config, image_lut))
    save_config(parsed_args.output_dir, "sample_improved.csv", final_config)

    image = imread("data/image.png")
    plot_traj(final_points, image)


if __name__ == "__main__":
    main()
