import os
import random

import numpy as np

from .cost import color_cost
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


def travel_map(df_image, output_dir, epsilon=0.0):

    path_result = []

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

    # Output header and origin configuration:
    f = open(os.path.join(output_dir, "submission.csv"), "w")
    print("configuration", file=f)  # header
    print(config_to_string(origin), file=f)  # origin configuration
    path_result.append(origin.copy())

    # Output arrows for visualization:
    a = open(os.path.join(output_dir, "arrows.csv"), "w")
    print("x,y,dx,dy", file=a)  # header

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

        # storage = TopKStorage()

        # Single-link step:
        for i in range(len(origin)):  # for each arm link
            for d in [-1, 1]:  # for each direction
                # Rotate link and get new position and vertical displacement:
                config2 = rotate(np.array(config), i, d)
                pos = get_position(np.array(config2))
                dy = pos[1] - base[1]

                # Convert from cartesian to array coordinates and measure cost:
                pos_arr = (pos[0] + radius, pos[1] + radius)
                cost2 = 1 + color_cost(base_arr, pos_arr, image)

                # Must move down unless impossible:
                if (
                    unvisited[pos_arr]
                    and cost2 < cost
                    and (dy < 0 or (dy >= 0 and below == 0))
                ):
                    # storage.push(config2.copy(), cost2)
                    config_next = config2.copy()
                    cost = cost2
                    found = True

        if below == 0:
            # Double-link step:
            for i in range(len(origin) - 1):
                for d1 in [-1, 1]:
                    for j in range(i + 1, len(origin)):
                        for d2 in [-1, 1]:
                            # Rotate two separate links, get position and vertical displacement:
                            config2 = rotate(np.array(config), i, d1)
                            config2 = rotate(np.array(config2), j, d2)
                            pos = get_position(np.array(config2))
                            dy = pos[1] - base[1]

                            # Convert from cartesian to array coordinates and measure cost:
                            pos_arr = (pos[0] + radius, pos[1] + radius)
                            cost2 = np.sqrt(2) + color_cost(base_arr, pos_arr, image)

                            # Must move down unless impossible:
                            if unvisited[pos_arr] and cost2 < cost and below == 0:
                                # storage.push(config2.copy(), cost2)
                                config_next = config2.copy()
                                cost = cost2
                                found = True

        # If an unvisited point was found, we are done for this step:
        if found:
            # if random.random() < epsilon and (not storage.empty()):
            #    config = storage.sample()
            # else:
            config = config_next.copy()
            pos = get_position(np.array(config))
            total -= 1

            # Print configuration and arrows:
            print(config_to_string(config), file=f)
            path_result.append(config.copy())
            print(
                base[0],
                ",",
                base[1],
                ",",
                pos[0] - base[0],
                ",",
                pos[1] - base[1],
                file=a,
            )

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
            if random.random() < epsilon and (not storage.empty()):
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

                # Print configuration and arrows:
                print(config_to_string(config), file=f)
                path_result.append(config.copy())
                print(
                    base[0],
                    ",",
                    base[1],
                    ",",
                    pos[0] - base[0],
                    ",",
                    pos[1] - base[1],
                    file=a,
                )
                base = pos

    # Return to origin:
    base = get_position(np.array(config))
    path = get_path_to_configuration(np.array(config), np.array(origin))[1:]

    # Output return trajectory:
    for config in path:
        pos = get_position(np.array(config))

        # Print configuration and arrows:
        print(config_to_string(config), file=f)
        path_result.append(config.copy())
        print(
            base[0], ",", base[1], ",", pos[0] - base[0], ",", pos[1] - base[1], file=a
        )
        base = pos

    # Close output files:
    f.close()
    a.close()

    return path_result
