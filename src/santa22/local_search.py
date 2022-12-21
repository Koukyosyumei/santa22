import random

import numpy as np
from tqdm import tqdm

from .cost import evaluate_config, total_cost
from .utils import get_path_to_configuration


def local_search_2opt(config, image_lut, max_itr=10):
    offset = random.randint(1, 10)
    best_score = evaluate_config(np.array(config), image_lut)
    print("initial score is ", best_score)

    for _ in tqdm(range(max_itr)):
        i = random.randint(0, len(config) - (3 + offset))
        c_1 = config[i]
        c_4 = config[i + 2 + offset]
        c_32 = list(reversed(config[i + 1 : i + 2 + offset]))

        c_13 = list(get_path_to_configuration(np.array(c_1), np.array(c_32[0])))
        c_24 = list(get_path_to_configuration(np.array(c_32[-1]), np.array(c_4)))

        # assert config[i].tolist() == c_13[0].tolist()
        # assert c_13[-1].tolist() == c_32[0].tolist()
        # assert c_32[-1].tolist() == c_24[0].tolist()
        # assert c_24[-1].tolist() == config[i + 2 + offset].tolist()

        config_new = config[:i] + c_13 + c_32[1:] + c_24[1:] + config[i + 3 + offset :]
        current_score = evaluate_config(np.array(config_new), image_lut)
        if current_score < best_score:
            best_score = current_score
            config = config_new

    print("improved score is ", best_score)
    return config
