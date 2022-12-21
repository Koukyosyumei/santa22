import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd

from santa22.greedy import travel_map
from santa22.local_search import local_search_2opt
from santa22.utils import save_config


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
        "--epsilon",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=42,
        type=int,
    )
    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    random.seed(parsed_args.seed)

    df = pd.read_csv(os.path.join(parsed_args.data_dir, "image.csv"))
    image_lut = df_to_imagelut(df)

    if parsed_args.initial_path is None:
        path_result = travel_map(df, parsed_args.output_dir, parsed_args.epsilon)
        with open(
            os.path.join(parsed_args.output_dir, "initial_path.pickle"), mode="wb"
        ) as f:
            pickle.dump(path_result, f)
    else:
        with open(parsed_args.initial_path, mode="rb") as f:
            path_result = pickle.load(f)

    path_result_improved = local_search_2opt(np.array(path_result), image_lut, 1000)
    save_config(parsed_args.output_dir, "sample_improved.csv", path_result_improved)


if __name__ == "__main__":
    main()
