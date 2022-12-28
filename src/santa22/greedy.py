import os
import random

import numpy as np
from numba import njit
from tqdm import tqdm

from .cost import color_cost
from .unionfind import UnionFind
from .utils import (
    config_to_string,
    get_path_to_configuration,
    get_path_to_point,
    get_position,
    rotate,
)


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
                and cost_cur < cost
                and (dy < 0 or (dy >= 0 and below == 0))
            ):
                config_next = config_cur.copy()
                cost = cost_cur
                found = True

    return config_next, cost, found


@njit
def double_link_step(
    origin, config, base, base_arr, radius, image, unvisited, cost, found
):
    config_next = config.copy()
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
                        config_next = config_cur.copy()
                        cost = cost_cur
                        found = True
                        update = True

    return update, config_next, cost, found


@njit
def triple_link_step(
    origin, config, base, base_arr, radius, image, unvisited, cost, found
):
    config_next = config.copy()
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
                                config_next = config_cur.copy()
                                cost = cost_cur
                                found = True
                                update = True

    return update, config_next, cost, found


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

            """
            for i in range(len(origin) - 1):
                for d1 in [-1, 1]:
                    for j in range(i + 1, len(origin)):
                        for d2 in [-1, 1]:
                            # Rotate two separate links, get position and vertical displacement:
                            config_cur = rotate(np.array(config), i, d1)
                            config_cur = rotate(config_cur, j, d2)
                            pos = get_position(config_cur)
                            dy = pos[1] - base[1]

                            # Convert from cartesian to array coordinates and measure cost:
                            pos_arr = (pos[0] + radius, pos[1] + radius)
                            cost_cur = np.sqrt(2) + color_cost(base_arr, pos_arr, image)

                            # Must move down unless impossible:
                            if unvisited[pos_arr] and cost_cur < cost:
                                config_next = config_cur.copy()
                                cost = cost_cur
                                found = True
            """

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
            storage = TopKStorage()

            for i in range(side):
                for j in range(side):
                    if unvisited[(i, j)]:

                        # Measure the distance to the current point and choose the nearest one:
                        distance2 = np.sqrt(
                            (base_arr[0] - i) ** 2 + (base_arr[1] - j) ** 2
                        )
                        if distance2 < distance:
                            storage.push((i - radius, j - radius), distance2)
                            point = (i - radius, j - radius)
                            distance = distance2

            # Go to the nearest unvisited point:
            if random.random() < epsilon:
                point = storage.sample()
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
