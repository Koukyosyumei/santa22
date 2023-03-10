import os
import pickle
import random

from tqdm import tqdm

from .cost import evaluate_config
from .tsp import double_bridge, three_opt, two_opt, two_opt_greedy
from .utils import run_remove

offset_choice = list(range(1, 21))
offset_choice_weight = [
    0.30095489,
    0.18505104,
    0.04412249,
    0.08791571,
    0.08034244,
    0.05795193,
    0.04543958,
    0.02041488,
    0.01448798,
    0.0250247,
    0.02765887,
    0.02403688,
    0.0154758,
    0.01679289,
    0.00889035,
    0.01053671,
    0.00724399,
    0.01514653,
    0.00823181,
    0.00428054,
]


def local_search(config, image_lut, output_dir, max_itr=10, t_start=0.3, t_end=0.001):
    config = run_remove(config)
    config = run_remove(config)
    removed_duplicate_score = evaluate_config(config, image_lut)
    print("remove duplicates: ", removed_duplicate_score)
    with open(
        os.path.join(output_dir, f"initial_path_{removed_duplicate_score}.pickle"),
        mode="wb",
    ) as f:
        pickle.dump(config, f)

    """
    config = two_opt_greedy(
        config,
        image_lut,
        random.sample(list(range(1, len(config))), len(config) - 1),
        2,
    )
    """

    initial_score = evaluate_config(config, image_lut)
    current_score = initial_score
    best_score = initial_score
    print("greedy two-opt: ", current_score)

    f = open("offset.csv", "w")

    tolerance_cnt = 0

    for itr in tqdm(range(max_itr)):

        if itr % 2 == 0:
            offset_1 = random.choices(offset_choice, weights=offset_choice_weight)[0]
            offset_2 = random.choices(offset_choice, weights=offset_choice_weight)[0]
            config_new, improve_score, improve_flag = three_opt(
                config, offset_1, offset_2, image_lut, t_start, t_end, itr, max_itr
            )
        else:
            offset = random.choices(offset_choice, weights=offset_choice_weight)[0]
            config_new, improve_score, improve_flag = two_opt(
                config, offset, image_lut, t_start, t_end, itr, max_itr
            )

        if improve_flag:
            print(f"{itr},{offset}", file=f)
            tolerance_cnt = 0
        else:
            tolerance_cnt += 1

        if improve_score != 0:
            current_score = current_score + improve_score
            config = config_new

        if current_score < best_score:
            best_config = config.copy()
            best_score = current_score

        if tolerance_cnt > 500000:
            config = double_bridge(config)
            tolerance_cnt = 0

        if (itr + 1) % 500000 == 0:
            config = run_remove(config)
            current_score = evaluate_config(config, image_lut)
            print(current_score)

    f.close()

    best_config = run_remove(config)
    best_config = run_remove(best_config)
    final_score = evaluate_config(best_config, image_lut)

    print("improved score is ", final_score)
    return best_config, initial_score > final_score
