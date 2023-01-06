import random

import numpy as np
from numba import njit
from tqdm import tqdm

from .cost import color_cost
from .utils import get_path_to_configuration, get_path_to_point, get_position, rotate


class TopKStorage:
    def __init__(self, k=3):
        self.k = k
        self.list_obj = []
        self.list_score = []

    def push(self, obj, score):
        if len(self.list_obj) < self.k:
            self.list_obj.append(obj)
            self.list_score.append(score)
        else:
            if score < np.max(self.list_score):
                max_idx = np.argmax(self.list_score)
                self.list_obj.pop(max_idx)
                self.list_obj.append(obj)
                self.list_score.pop(max_idx)
                self.list_score.append(score)

    def clear(self):
        self.list_obj = []
        self.list_score = []

    def sample(self):
        return random.choice(self.list_obj)

    def empty(self):
        return len(self.list_obj) == 0


@njit
def single_link_step(
    origin, config, base, base_arr, radius, image, unvisited, below, cost, found
):
    config_next = config.copy()
    cost_store = [config.copy()]
    cost_store_init = True

    for i in range(len(origin)):  # for each arm link
        for d in [-1, 1]:  # for each direction
            # Rotate link and get new position and vertical displacement:
            config_cur = rotate(config, i, d)
            pos = get_position(config_cur)
            dy = pos[1] - base[1]

            # Convert from cartesian to array coordinates and measure cost:
            pos_arr = (pos[0] + radius, pos[1] + radius)
            cost_cur = 1 + color_cost(base_arr, pos_arr, image)

            # Must move down unless impossible:
            if (
                unvisited[pos_arr]
                and cost_cur <= cost
                and (dy < 0 or (dy >= 0 and below == 0))
            ):
                if cost_store_init:
                    cost_store.pop()
                    cost_store_init = False

                if cost_cur == cost:
                    cost_store.append(config_cur.copy())
                else:
                    cost_store = [config_cur.copy()]

                cost = cost_cur
                found = True

    if not cost_store_init:
        config_next = cost_store[random.randint(0, len(cost_store) - 1)]

    return config_next, cost, found


@njit
def double_link_step(
    origin, config, base, base_arr, radius, image, unvisited, cost, found
):
    config_next = config.copy()
    cost_store = [config.copy()]
    cost_store_init = True
    update = False

    for i in range(len(origin) - 1):
        for d1 in [-1, 1]:
            for j in range(i + 1, len(origin)):
                for d2 in [-1, 1]:
                    # Rotate two separate links, get position and vertical displacement:
                    config_cur = rotate(config, i, d1)
                    config_cur = rotate(config_cur, j, d2)
                    pos = get_position(config_cur)
                    dy = pos[1] - base[1]

                    # Convert from cartesian to array coordinates and measure cost:
                    pos_arr = (pos[0] + radius, pos[1] + radius)
                    cost_cur = np.sqrt(2) + color_cost(base_arr, pos_arr, image)

                    # Must move down unless impossible:
                    if unvisited[pos_arr] and cost_cur < cost:
                        if cost_store_init:
                            cost_store.pop()
                            cost_store_init = False

                        if cost_cur == cost:
                            cost_store.append(config_cur.copy())
                        else:
                            cost_store = [config_cur.copy()]

                        cost = cost_cur
                        found = True
                        update = True

    if not cost_store_init:
        config_next = cost_store[random.randint(0, len(cost_store) - 1)]

    return update, config_next, cost, found


@njit
def triple_link_step(
    origin, config, base, base_arr, radius, image, unvisited, cost, found
):
    config_next = config.copy()
    cost_store = [config.copy()]
    cost_store_init = True
    update = False

    for i in range(len(origin) - 1):
        for d1 in [-1, 1]:
            for j in range(i + 1, len(origin)):
                for d2 in [-1, 1]:
                    for k in range(j + 1, len(origin)):
                        for d3 in [-1, 1]:
                            # Rotate three separate links, get position and vertical displacement:
                            config_cur = rotate(config, i, d1)
                            config_cur = rotate(config_cur, j, d2)
                            config_cur = rotate(config_cur, k, d3)
                            pos = get_position(config_cur)
                            dy = pos[1] - base[1]

                            # Convert from cartesian to array coordinates and measure cost:
                            pos_arr = (pos[0] + radius, pos[1] + radius)
                            cost_cur = np.sqrt(3) + color_cost(base_arr, pos_arr, image)

                            # Must move down unless impossible:
                            if unvisited[pos_arr] and cost_cur < cost:
                                if cost_store_init:
                                    cost_store.pop()
                                    cost_store_init = False

                                if cost_cur == cost:
                                    cost_store.append(config_cur.copy())
                                else:
                                    cost_store = [config_cur.copy()]

                                cost = cost_cur
                                found = True
                                update = True

    if not cost_store_init:
        config_next = cost_store[random.randint(0, len(cost_store) - 1)]

    return update, config_next, cost, found


@njit
def four_link_step(
    origin, config, base, base_arr, radius, image, unvisited, cost, found
):
    config_next = config.copy()
    cost_store = [config.copy()]
    cost_store_init = True
    update = False

    for i in range(len(origin) - 1):
        for d1 in [-1, 1]:
            config_cur_1 = rotate(config, i, d1)
            for j in range(i + 1, len(origin)):
                for d2 in [-1, 1]:
                    config_cur_2 = rotate(config_cur_1, j, d2)
                    for k in range(j + 1, len(origin)):
                        for d3 in [-1, 1]:
                            config_cur_3 = rotate(config_cur_2, k, d3)
                            for l in range(j + 1, len(origin)):
                                for d4 in [-1, 1]:
                                    # Rotate three separate links, get position and vertical displacement:
                                    config_cur = rotate(config_cur_3, l, d4)
                                    pos = get_position(config_cur)
                                    dy = pos[1] - base[1]

                                    # Convert from cartesian to array coordinates and measure cost:
                                    pos_arr = (pos[0] + radius, pos[1] + radius)
                                    cost_cur = np.sqrt(4) + color_cost(
                                        base_arr, pos_arr, image
                                    )

                                    # Must move down unless impossible:
                                    if unvisited[pos_arr] and cost_cur < cost:
                                        if cost_store_init:
                                            cost_store.pop()
                                            cost_store_init = False

                                        if cost_cur == cost:
                                            cost_store.append(config_cur.copy())
                                        else:
                                            cost_store = [config_cur.copy()]

                                        cost = cost_cur
                                        found = True
                                        update = True

    if not cost_store_init:
        config_next = cost_store[random.randint(0, len(cost_store) - 1)]

    return update, config_next, cost, found


@njit
def twopart(n):
    return n & (n - 1) == 0


@njit
def find_near(side, unvisited, base_arr, radius, distance):
    # Go to the nearest unvisited point:

    k = 2
    list_obj = [np.array((-1000, -1000))]
    list_score = [100000.00]
    list_init = True

    for i in range(side):
        for j in range(side):
            if unvisited[(i, j)]:

                # Measure the distance to the current point and choose the nearest one:
                penalty = 1
                p = 0.16  # 0.1 - 79200
                if twopart(base_arr[0] - i):
                    penalty += p
                if twopart(base_arr[1] - j):
                    penalty += p
                distance2 = penalty * np.sqrt(
                    (base_arr[0] - i) ** 2 + (base_arr[1] - j) ** 2
                )

                if distance2 < distance:
                    point = (i - radius, j - radius)
                    distance = distance2

                    if list_init:
                        list_obj.pop(0)
                        list_score.pop(0)
                        list_init = False

                    if len(list_obj) < k:
                        list_obj.append(np.array((i - radius, j - radius)))
                        list_score.append(distance2)
                    else:
                        if distance2 < np.max(np.array(list_score)):
                            max_idx = np.argmax(np.array(list_score))
                            list_obj.pop(max_idx)
                            list_obj.append(np.array((i - radius, j - radius)))
                            list_score.pop(max_idx)
                            list_score.append(distance2)

    return point, list_obj, list_score


def travel_map(df_image, output_dir, epsilon=0.0):

    side = df_image.x.nunique()
    radius = df_image.x.max()
    image = df_image[["r", "g", "b"]].values.reshape(side, side, -1)

    # Flip X axis and transpose X-Y axes to simplify cartesian to array mapping:
    image = image[::-1, :, :]
    image = np.transpose(image, (1, 0, 2))

    # Prepare pixel travel map:
    unvisited = np.ones([side, side])  # one = unvisited pixel; 0 = visited pixel
    total = side * side - 1  # total number of pixels minus the origin
    origin = [
        (64, 0),
        (-32, 0),
        (-16, 0),
        (-8, 0),
        (-4, 0),
        (-2, 0),
        (-1, 0),
        (-1, 0),
    ]  # origin configuration
    config = origin.copy()  # future configuration

    result = [config]
    pbar = tqdm(total=total)
    # Continue until all locations have been visited:
    while total:

        # Optimization variables:
        cost = 1e6
        distance = 1e6
        found = False

        # Current configuration:
        base = get_position(np.array(config))
        base_arr = (base[0] + radius, base[1] + radius)
        unvisited[base_arr] = 0

        # Is the location one step below unvisited?
        if base[1] == -128:  # if we reached the bottom border
            below = 0
        else:
            below = unvisited[(base_arr[0], base_arr[1] - 1)]

        # Single-link step:
        config_next, cost, found = single_link_step(
            np.array(origin),
            np.array(config),
            base,
            base_arr,
            radius,
            image,
            unvisited,
            below,
            cost,
            found,
        )

        if below == 0:
            # Double-link step:
            update, tmp_config_next, tmp_cost, tmp_found = double_link_step(
                np.array(origin),
                np.array(config),
                base,
                base_arr,
                radius,
                image,
                unvisited,
                cost,
                found,
            )

            if update:
                config_next = tmp_config_next.copy()
                cost = tmp_cost
                found = tmp_found

            # Tripple-link step:
            update, tmp_config_next, tmp_cost, tmp_found = triple_link_step(
                np.array(origin),
                np.array(config),
                base,
                base_arr,
                radius,
                image,
                unvisited,
                cost,
                found,
            )

            if update:
                config_next = tmp_config_next.copy()
                cost = tmp_cost
                found = tmp_found

            # Four-link step:
            update, tmp_config_next, tmp_cost, tmp_found = four_link_step(
                np.array(origin),
                np.array(config),
                base,
                base_arr,
                radius,
                image,
                unvisited,
                cost,
                found,
            )

            if update:
                config_next = tmp_config_next.copy()
                cost = tmp_cost
                found = tmp_found

        # If an unvisited point was found, we are done for this step:
        if found:
            config = config_next.copy()
            pos = get_position(np.array(config))
            total -= 1
            pbar.update(1)
            result.append(config)

        # Otherwise, find the nearest unvisited point and go there ignoring the travel map:
        else:
            # Search every single pixel of the travel map for unvisited points:
            point, list_obj, _ = find_near(side, unvisited, base_arr, radius, distance)
            if random.random() < epsilon and len(list_obj) != 0:
                point = list_obj[random.randint(0, len(list_obj) - 1)]

            path = get_path_to_point(np.array(config), np.array(point))[1:]

            # Output shortest trajectory:
            for config in path:
                pos = get_position(np.array(config))
                pos_arr = (pos[0] + radius, pos[1] + radius)

                # Update the travel map:
                if unvisited[pos_arr]:
                    unvisited[pos_arr] = 0
                    total -= 1
                    pbar.update(1)

                result.append(config)

                base = pos

    pbar.close()

    # Return to origin:
    path = get_path_to_configuration(np.array(config), np.array(origin))[1:]

    result.extend(path)

    return result
