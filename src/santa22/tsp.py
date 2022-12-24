import math
import random

import numpy as np
from numba import njit

from .cost import evaluate_config
from .utils import get_path_to_configuration


@njit
def calc_threshold(improve, t_start, t_final, current_itr, max_itr):
    t = t_start + (t_final - t_start) * current_itr / max_itr
    return math.exp(improve / t)


@njit
def double_bridge(config):
    offset = 1
    i = random.randint(1, len(config) - (6 + 3 * offset))
    j = i + 1 + offset
    k = i + 2 + 2 * offset
    m = i + 3 + 3 * offset

    # AB - CD - EF - GH -> A-D C-F E-H G-B

    p_AD = get_path_to_configuration(config[i - 1], config[j])
    p_CF = get_path_to_configuration(config[j - 1], config[k])
    p_EH = get_path_to_configuration(config[k - 1], config[m])
    p_GB = get_path_to_configuration(config[m - 1], config[i])
    p_BI = get_path_to_configuration(config[i], config[m + 1])

    return np.concatenate(
        (
            config[: i - 1],
            p_AD,
            p_CF,
            p_EH,
            p_GB,
            p_BI[1:],
            config[m + 2 :],
        )
    )


@njit
def three_opt(config, offset, image_lut, t_start, t_end, itr, max_itr):
    # offset = 1  # TODO support larger offset

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
        return (
            np.concatenate(
                (
                    config[: i - 1],
                    p_AC,
                    config[i + 1 : j - 1][::-1],
                    p_BD,
                    config[j + 1 :],
                )
            ),
            -d0 + d1,
            True,
        )
    elif d0 > d2:
        # CDEF -> CEDF
        return (
            np.concatenate(
                (
                    config[: j - 1],
                    p_CE,
                    config[j + 1 : k - 1][::-1],
                    p_DF,
                    config[k + 1 :],
                )
            ),
            -d0 + d2,
            True,
        )
    elif d0 > d4:
        # ABCDEF -> AEDCBF
        return (
            np.concatenate(
                (
                    config[: i - 1],
                    p_EA[::-1],
                    config[j + 1 : k - 1][::-1],
                    config[j - 1 : j + 1][::-1],
                    config[i + 1 : j - 1][::-1],
                    p_FB[::-1],
                    config[k + 1 :],
                )
            ),
            -d0 + d4,
            True,
        )
    elif d0 > d3:
        # ABCDEF -> ADEBCF
        return (
            np.concatenate(
                (
                    config[: i - 1],
                    p_AD,
                    config[j + 1 : k - 1],
                    p_EB,
                    config[i + 1 : j - 1],
                    p_CF,
                    config[k + 1 :],
                )
            ),
            -d0 + d3,
            True,
        )
    elif random.random() < calc_threshold(d0 - d3, t_start, t_end, itr, max_itr):
        return (
            np.concatenate(
                (
                    config[: i - 1],
                    p_AD,
                    config[j + 1 : k - 1],
                    p_EB,
                    config[i + 1 : j - 1],
                    p_CF,
                    config[k + 1 :],
                )
            ),
            -d0 + d3,
            False,
        )

    return config, 0, False


@njit
def two_opt(config, offset, image_lut, t_start, t_end, itr, max_itr):
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

    if d0 > d1 or random.random() < calc_threshold(
        d0 - d1, t_start, t_end, itr, max_itr
    ):
        return (
            np.concatenate((config[:i], p_AC, p_CB[1:], p_BD[1:], config[j + 1 :])),
            -d0 + d1,
            d0 > d1,
        )

    return config, 0, False
