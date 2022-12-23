import math
import random

import numpy as np
from numba import njit
from tqdm import tqdm

from .cost import evaluate_config
from .utils import get_path_to_configuration, run_remove

offset_choice = [1, 2, 3, 4, 5, 6, 7]
offset_choice_weight_near = [0.7, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02]
offset_choice_weight_far = [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]


def four_opt(config, offset):
    i = random.randint(0, len(config) - (5 + 4 * offset))

    c_1 = config[i]
    c_2 = config[i + offset]

    c_3 = config[i + 1 + offset]
    c_4 = config[i + 1 + 2 * offset]

    c_5 = config[i + 2 + 2 * offset]
    c_6 = config[i + 2 + 3 * offset]

    c_7 = config[i + 3 + 3 * offset]
    c_8 = config[i + 3 + 4 * offset]
    c_9 = config[i + 4 + 4 * offset]

    c_14 = get_path_to_configuration(c_1, c_4)
    c_43 = config[i + 1 + offset : i + 2 + 2 * offset][::-1]
    c_36 = get_path_to_configuration(c_3, c_6)
    c_65 = config[i + 2 + 2 * offset : i + 3 + 3 * offset][::-1]
    c_58 = get_path_to_configuration(c_5, c_8)
    c_87 = config[i + 3 + 3 * offset : i + 4 + 4 * offset][::-1]
    c_72 = get_path_to_configuration(c_7, c_2)
    c_21 = config[i : i + 1 + offset][::-1]
    c_19 = get_path_to_configuration(c_1, c_9)

    config_new = np.concatenate(
        [
            config[:i],
            c_14,
            c_43[1:],
            c_36[1:],
            c_65[1:],
            c_58[1:],
            c_87[1:],
            c_72[1:],
            c_21[1:],
            c_19[1:],
            config[i + 5 + 4 * offset :],
        ]
    )

    return config_new


@njit
def calc_threshold(improve, t_start, t_final, current_itr, max_itr):
    t = t_start + (t_final - t_start) * current_itr / max_itr
    return math.exp(improve / t)


@njit
def two_opt(config, offset, image_lut):
    i = random.randint(0, len(config) - (3 + offset))

    c_32 = config[i + 1 : i + 2 + offset][::-1]
    c_13 = get_path_to_configuration(config[i], c_32[0])
    c_24 = get_path_to_configuration(c_32[-1], config[i + 2 + offset])

    # assert config[i].tolist() == c_13[0].tolist()
    # assert c_13[-1].tolist() == c_32[0].tolist()
    # assert c_32[-1].tolist() == c_24[0].tolist()
    # assert c_24[-1].tolist() == config[i + 2 + offset].tolist()

    config_new = np.concatenate(
        (config[:i], c_13, c_32[1:], c_24[1:], config[i + 3 + offset :])
    )

    improve = evaluate_config(
        np.concatenate((c_13, c_32[1:], c_24[1:])), image_lut
    ) - evaluate_config(config[i : i + 3 + offset], image_lut)

    return config_new, improve


def local_search(config, image_lut, max_itr=10, t_start=0.3, t_end=0.01):
    config = run_remove(config)
    initial_score = evaluate_config(config, image_lut)
    best_score = initial_score
    print("initial score is ", best_score)

    for itr in tqdm(range(max_itr)):
        if itr < 1000000:
            offset = random.choices(offset_choice, weights=offset_choice_weight_near)[0]
        else:
            offset = random.choices(offset_choice, weights=offset_choice_weight_far)[0]

        config_new, improve = two_opt(config, offset, image_lut)

        if improve < 0 or random.random() < calc_threshold(
            improve * -1, t_start, t_end, itr, max_itr
        ):
            best_score = best_score + improve
            config = config_new

        if itr % 10000 == 0:
            config = run_remove(config)
            print(best_score)

    config = run_remove(config)
    final_score = evaluate_config(config, image_lut)

    print("improved score is ", final_score)
    return config, initial_score > final_score
