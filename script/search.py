import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd

from santa22.cost import evaluate_config
from santa22.greedy import travel_map
from santa22.local_search import local_search
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

    if parsed_args.initial_path is None:
        path_result = travel_map(df, parsed_args.output_dir, parsed_args.epsilon_greedy)
        with open(
            os.path.join(parsed_args.output_dir, "initial_path.pickle"), mode="wb"
        ) as f:
            pickle.dump(path_result, f)
    else:
        with open(parsed_args.initial_path, mode="rb") as f:
            path_result = pickle.load(f)

    print(
        f"initial_score: {evaluate_config(np.array(path_result), image_lut)} (Random Seed = {parsed_args.seed})"
    )

    path_result_improved, updated_flag = local_search(
        np.array(path_result),
        image_lut,
        parsed_args.output_dir,
        parsed_args.max_itr,
        parsed_args.t_start,
        parsed_args.t_end,
    )
    save_config(parsed_args.output_dir, "sample_improved.csv", path_result_improved)

    if updated_flag:
        with open(
            os.path.join(parsed_args.output_dir, "initial_path.pickle"), mode="wb"
        ) as f:
            pickle.dump(path_result_improved, f)


if __name__ == "__main__":
    main()
