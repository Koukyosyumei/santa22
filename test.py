import numpy as np
import pandas as pd


def judge_straight(points, i):
    return points[i - 1][0] == points[i][0] or points[i - 1][1] == points[i][1]


def fix_zigzag(points):
    i = 1
    len_points = len(points)
    prev_straight = False

    while i < len_points:
        if judge_straight(i):
            i += 1
            prev_straight = True
        else:
            i += 1
            prev_straight = False
