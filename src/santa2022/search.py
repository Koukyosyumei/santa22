
import numpy as np
from .evaluator import evaluate_config
from .cost import total_cost
from .path import points_to_path
import random


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


def explore(current_solution_, transitions):
    try:
        neighbouring_solution_ = current_solution_.copy()

        # Currently, the exploration is performed by swapping the place of two adjacent points; you can come up with other methods to explore the neighbourhood of a solution
        offset = 1
        i_ = random.randint(0, len(neighbouring_solution_) - (offset + 1))

        if i_ not in transitions:
            transitions.add(i_)

            t = neighbouring_solution_[i_]
            neighbouring_solution_[i_] = current_solution_[i_ + offset]
            neighbouring_solution_[i_ + offset] = t

            # Compute the cost of transition to the neighbouring solution
            neighbouring_solutions_cost_ = evaluate_config(
                points_to_path(np.array(neighbouring_solution_)))

            return neighbouring_solutions_cost_, neighbouring_solution_
        else:
            return 10 ** 6, current_solution_
    except Exception as error_:
        print(error_)
        return 10 ** 6, current_solution_
