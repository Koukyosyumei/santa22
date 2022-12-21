import random

import numpy as np
from tqdm import tqdm

from .cost import evaluate_config
from .utils import get_path_to_configuration

offset_choice = [1, 2, 3]
offset_choice_weight = [
    0.7,
    0.2,
    0.1,
]


def local_search_2opt(config, image_lut, max_itr=10):
    initial_score = evaluate_config(config, image_lut)
    best_score = initial_score
    print("initial score is ", best_score)

    for _ in tqdm(range(max_itr)):
        offset = random.choices(offset_choice, weights=offset_choice_weight)[0]
        i = random.randint(0, len(config) - (3 + offset))
        c_1 = config[i]
        c_4 = config[i + 2 + offset]
        c_32 = config[i + 1 : i + 2 + offset][::-1]

        c_13 = get_path_to_configuration(c_1, c_32[0])
        c_24 = get_path_to_configuration(c_32[-1], c_4)

        # assert config[i].tolist() == c_13[0].tolist()
        # assert c_13[-1].tolist() == c_32[0].tolist()
        # assert c_32[-1].tolist() == c_24[0].tolist()
        # assert c_24[-1].tolist() == config[i + 2 + offset].tolist()

        config_new = np.concatenate(
            [config[:i], c_13, c_32[1:], c_24[1:], config[i + 3 + offset :]]
        )
        current_score = evaluate_config(config_new, image_lut)
        if current_score < best_score:
            print(current_score, offset)
            best_score = current_score
            config = config_new

    print("improved score is ", best_score)
    return config, initial_score > best_score
