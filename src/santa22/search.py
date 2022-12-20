import random

import numpy as np
import tqdm

from .cost import total_cost
from .evaluator import evaluate_config
from .path import points_to_path
from .utils import check_point


def make_starting_solution(image_):
    # Generate points
    points_ = []
    flag = True
    for split in range(2):
        for i_ in reversed(range(257)) if split % 2 == 0 else range(257):
            if not flag:
                for j in range(128 * split, 128 + 129 * split):
                    points_.append((j - 128, i_ - 128))
            else:
                for j in reversed(range(128 * split, 128 + 129 * split)):
                    points_.append((j - 128, i_ - 128))
            flag = not flag
        flag = False

    # Generate path
    points_ = np.array(points_)

    # Make path
    path_ = points_to_path(points_)

    # Compute cost
    cost_ = total_cost(path_, image_)

    return points_, path_, cost_


def explore(current_solution_, transitions, image_lut):
    try:
        neighbouring_solution_ = current_solution_.copy()

        # Currently, the exploration is performed by swapping
        # the place of two adjacent points;
        # you can come up with other methods to explore
        # the neighbourhood of a solution
        offset = 1
        i_ = random.randint(0, len(neighbouring_solution_) - (offset + 1))

        if i_ not in transitions:
            transitions.add(i_)

            t = neighbouring_solution_[i_]
            neighbouring_solution_[i_] = current_solution_[i_ + offset]
            neighbouring_solution_[i_ + offset] = t

            # Compute the cost of transition to the neighbouring solution
            neighbouring_solutions_cost_ = evaluate_config(
                points_to_path(np.array(neighbouring_solution_)), image_lut
            )

            return neighbouring_solutions_cost_, neighbouring_solution_
        else:
            return 10**6, current_solution_
    except Exception as error_:
        print(error_)
        return 10**6, current_solution_


def iterate_search(
    current_solution,
    current_solutions_cost,
    image_lut,
    max_iterations=1000,
    check_pointing_interval=200,
):
    transitions = set()

    for iterations in tqdm.tqdm(range(max_iterations)):
        iterations += 1
        neighbouring_solutions_cost, neighbouring_solution = explore(
            current_solution, transitions, image_lut
        )
        if neighbouring_solutions_cost < current_solutions_cost:
            print(
                f"--> Found a better solutions at the {iterations}th interation;"
                f" the improvement by transitioning to the better solution was:"
                f" {current_solutions_cost - neighbouring_solutions_cost}"
            )
            current_solutions_cost = neighbouring_solutions_cost
            current_solution = neighbouring_solution
            transitions = set()
        else:
            pass

        if iterations % 100 == 0:
            print(f"--> {iterations} neighbours have been explored")

        # save the best points every 1k iterations
        if iterations % check_pointing_interval == 0:
            print(
                f"--> Making a check-point of the current best solution with cost: {current_solutions_cost:.3f}"
            )
            check_point(current_solutions_cost, current_solution)

    print(
        f"Search ended after {max_iterations} iterations;"
        f" the cost of the new best solution found during the search was: {current_solutions_cost:.3f}"
    )

    return current_solution, current_solutions_cost
