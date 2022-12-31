from functools import *
from itertools import *

import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from numba import njit
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

N = 257
NN = N * N
radius = 128


def df_to_image(df_image):
    side = int(len(df_image) ** 0.5)  # assumes a square image
    return df_image.set_index(["x", "y"]).to_numpy().reshape(side, side, -1)


@njit
def hash_to_xy(i):
    return i // N, i % N


@njit
def xy_to_hash(x, y):
    return x * N + y


@njit
def get_cost(i, j, image):
    ix, iy = hash_to_xy(i)
    jx, jy = hash_to_xy(j)
    d = np.abs(ix - jx) + np.abs(iy - jy)
    return np.sum(np.abs(image[ix, iy] - image[jx, jy])) * 3 + np.sqrt(d)


@njit
def calc_dist_between_hashed_points(i, j):
    ix, iy = hash_to_xy(i)
    jx, jy = hash_to_xy(j)
    d = np.sqrt(np.abs(ix - jx) ** 2 + np.abs(iy - jy) ** 2)
    return d


def generate_adj_mat(image):
    mat = dok_matrix((NN, NN), dtype=np.float64)

    for i in tqdm.trange(NN):
        x, y = hash_to_xy(i)
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                nx, ny = x + dx, y + dy
                if np.abs(dx) + np.abs(dy) > 8:
                    continue
                if nx < 0 or N <= nx or ny < 0 or N <= ny:
                    continue
                j = xy_to_hash(nx, ny)
                cost = get_cost(i, j, image)
                mat[i, j] = cost

    return mat


def get_mst_edges(mat):
    mst = minimum_spanning_tree(mat)
    edges = list(zip(*mst.nonzero()))
    return edges


@njit
def get_next_point(to_points, image, cur_point, used_hashed_points):
    dist_to_points = [1e6 * 1.0]
    for p in to_points:
        dist_to_points.append(get_cost(p, cur_point, image))
    next_points_candidates = to_points[np.argsort(np.array(dist_to_points))]

    update = False
    for p in next_points_candidates:
        if used_hashed_points[p] == 0:
            next_point = p
            update = True
            break

    if not update:
        best_dist = 1e8
        for p in range(NN):
            if used_hashed_points[p] == 0:
                tmp_dist = get_cost(p, cur_point, image)
                if tmp_dist < best_dist:
                    best_p = p
                    tmp_dist = best_dist
        next_point = best_p

    return next_point


def get_hashed_points_order(edges, image):
    start_hashed_point = xy_to_hash(radius, radius)

    path_hashed_points = [start_hashed_point]
    used_hashed_points = np.zeros(NN)
    used_hashed_points[start_hashed_point] = 1

    total = NN - 1
    cur_point = start_hashed_point

    adj_edges = {i: [] for i in range(NN)}

    for e in edges:
        adj_edges[e[0]].append(e[1])
        adj_edges[e[1]].append(e[0])

    while total:

        to_points = np.array(adj_edges[cur_point])
        cur_point = get_next_point(to_points, image, cur_point, used_hashed_points)

        total = total - 1
        used_hashed_points[cur_point] = 1
        path_hashed_points.append(cur_point)

        if total % 1000 == 0:
            print(total)

    return path_hashed_points


def plot_mst(edges, image):
    lines = []
    for i, j in edges:
        ix, iy = hash_to_xy(i)
        jx, jy = hash_to_xy(j)
        # d = np.sqrt(np.abs(ix - jx) ** 2 + np.abs(iy - jy) ** 2)
        # if d >= 1.8:
        lines.append([[iy - radius, radius - ix], [jy - radius, radius - jx]])

    lc = mc.LineCollection(lines, colors="b")

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.add_collection(lc)

    ax.matshow(image, extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5))
    ax.grid(None)

    ax.autoscale()
    plt.savefig("mst.png")
