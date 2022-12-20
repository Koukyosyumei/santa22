import argparse
import os

import pandas as pd

from santa22.cost import total_cost
from santa22.travel import travel_map


def df_to_image(df):
    side = int(len(df) ** 0.5)  # assumes a square image
    return df.set_index(["x", "y"]).to_numpy().reshape(side, side, -1)


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
    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    df = pd.read_csv(os.path.join(parsed_args.data_dir, "image.csv"))
    image = df_to_image(df)

    path_result = travel_map(df, parsed_args.output_dir)

    print(total_cost(path_result, image))


if __name__ == "__main__":
    main()
