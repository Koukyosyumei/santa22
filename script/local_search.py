import argparse
import io
import os
import pickle

import numpy as np
import pandas as pd

from santa22 import iterate_search, make_starting_solution
from santa22.path import points_to_path
from santa22.utils import check_point, config_to_string, imread


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
        "-i",
        "--max_iterations",
        default=1000,
        type=int,
    )
    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    data_dir = parsed_args.data_dir
    output_dir = parsed_args.output_dir
    max_iterations = parsed_args.max_iterations

    image = imread(os.path.join(data_dir, "image.png"))
    print(image.shape)

    print(f"Starting search with max iterations: {max_iterations}")

    # Load the current solution and its cost, otherwise,
    # use the solution generated by running
    # 'make_starting_solution' as the starting solution
    try:
        with io.open(
            os.path.join(output_dir, "current_solution_data.pkl"), "rb"
        ) as in_file:
            saved_solution = pickle.load(in_file)
    except Exception as error:
        print(error)
        print(
            "No saved solution found, using the solution made by\
              'make_starting_solution' as the starting solution"
        )
        (
            starting_solutions_points,
            starting_solutions_path,
            starting_solutions_cost,
        ) = make_starting_solution(image)
        current_solution = [tuple(x) for x in starting_solutions_points]
        current_solutions_cost = starting_solutions_cost
        # plot_traj(starting_solutions_points, image)
        print(f"Loaded the starting solution with cost: {starting_solutions_cost:.3f}")
    else:
        current_solution = saved_solution["solution"]
        current_solutions_cost = saved_solution["cost"]
        print(
            f"Loaded the solution from the last search with cost: {current_solutions_cost:.3f}"
        )

    current_solution, current_solutions_cost = iterate_search(
        current_solution, current_solutions_cost, max_iterations
    )

    check_point(current_solutions_cost, current_solution)

    optimized_path = points_to_path(np.array(current_solution))
    submission_opt = pd.Series(
        [config_to_string(config) for config in optimized_path], name="configuration"
    )
    submission_opt.to_csv(
        os.path.join(output_dir, "submission_optimized.csv"), index=False
    )


if __name__ == "__main__":
    main()
