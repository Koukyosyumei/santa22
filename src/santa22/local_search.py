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
def three_opt(config, image_lut):
    offset = 1  # TODO support larger offset

    i = random.randint(1, len(config) - (3 + 2 * offset))
    j = i + offset + 1
    k = j + offset + 1

    d_AB = evaluate_config(config[i - 1 : i + 1], image_lut)
    d_CD = evaluate_config(config[j - 1 : j + 1], image_lut)
    d_EF = evaluate_config(config[k - 1 : k + 1], image_lut)

    p_AC = get_path_to_configuration(config[i - 1], config[j - 1])
    p_BD = get_path_to_configuration(config[i], config[j])
    p_CE = get_path_to_configuration(config[j - 1], config[k - 1])
    p_DF = get_path_to_configuration(config[j], config[k])
    p_AD = get_path_to_configuration(config[i - 1], config[j])
    p_EB = get_path_to_configuration(config[k - 1], config[i])
    p_CF = get_path_to_configuration(config[j - 1], config[k])
    p_FB = get_path_to_configuration(config[k], config[i])
    p_EA = get_path_to_configuration(config[k - 1], config[i - 1])

    d_AC = evaluate_config(p_AC, image_lut)
    d_BD = evaluate_config(p_BD, image_lut)
    d_CE = evaluate_config(p_CE, image_lut)
    d_DF = evaluate_config(p_DF, image_lut)
    d_AD = evaluate_config(p_AD, image_lut)
    d_EB = evaluate_config(p_EB, image_lut)
    d_CF = evaluate_config(p_CF, image_lut)
    d_FB = evaluate_config(p_FB, image_lut)
    d_EA = evaluate_config(p_EA, image_lut)

    d0 = d_AB + d_CD + d_EF
    d1 = d_AC + d_BD + d_EF
    d2 = d_AB + d_CE + d_DF
    d3 = d_AD + d_EB + d_CF
    d4 = d_FB + d_CD + d_EA

    if d0 > d1:
        # ABCD -> ACBD
        return np.concatenate((config[: i - 1], p_AC, p_BD, config[j + 1 :])), -d0 + d1
    elif d0 > d2:
        # CDEF -> CEDF
        return np.concatenate((config[: j - 1], p_CE, p_DF, config[k + 1 :])), -d0 + d2
    elif d0 > d4:
        # ABCDEF -> AEDCBF
        return (
            np.concatenate(
                (
                    config[: i - 1],
                    p_EA[::-1],
                    config[j - 1 : j + 1][::-1],
                    p_FB[::-1],
                    config[k + 1 :],
                )
            ),
            -d0 + d4,
        )
    elif d0 > d3:
        # ABCDEF -> ADEBCF
        return (
            np.concatenate((config[: i - 1], p_AD, p_EB, p_CF, config[k + 1 :])),
            -d0 + d3,
        )

    return config, 0


@njit
def two_opt(config, offset, image_lut):
    i = random.randint(1, len(config) - (3 + offset))
    j = i + 1 + offset

    p_AC = get_path_to_configuration(config[i - 1], config[j - 1])
    p_BD = get_path_to_configuration(config[i], config[j])

    d_AB = evaluate_config(config[i - 1 : i + 1], image_lut)
    d_CD = evaluate_config(config[j - 1 : j + 1], image_lut)
    d_AC = evaluate_config(p_AC, image_lut)
    d_BD = evaluate_config(p_BD, image_lut)

    d0 = d_AB + d_CD
    d1 = d_AC + d_BD

    p_CB = config[i:j][::-1]

    if d0 > d1:
        return (
            np.concatenate((config[:i], p_AC, p_CB[1:], p_BD[1:], config[j + 1 :])),
            -d0 + d1,
        )

    return config, 0


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

        if itr % 3 == 0:
            config_new, improve = three_opt(config, image_lut)
        else:
            config_new, improve = two_opt(config, offset, image_lut)

        if improve < 0 or (
            improve > 0
            and random.random()
            < calc_threshold(improve * -1, t_start, t_end, itr, max_itr)
        ):
            best_score = best_score + improve
            config = config_new

        if itr % 500000 == 0:
            config = run_remove(config)
            print(best_score)

    config = run_remove(config)
    final_score = evaluate_config(config, image_lut)

    print("improved score is ", final_score)
    return config, initial_score > final_score
