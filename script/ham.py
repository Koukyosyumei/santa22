import os
import pickle
from functools import *
from itertools import *
from pathlib import Path

import cv2
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
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


def imread(path):
    if isinstance(path, Path):
        path = path.as_posix()
    return cv2.imread(path)[:, :, ::-1] / 255


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
    print(f"total cost : {mst.sum()}")
    edges = list(zip(*mst.nonzero()))
    return mst, edges


@njit
def get_next_point(to_points, image, cur_point, used_hashed_points):
    dist_to_points = [1e6 * 1.0]
    for p in to_points:
        dist_to_points.append(get_cost(p, cur_point, image))
    dist_to_points = dist_to_points[1:]
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
                tmp_dist = calc_dist_between_hashed_points(p, cur_point)
                # get_cost(p, cur_point, image)
                if tmp_dist < best_dist:
                    best_p = p
                    best_dist = tmp_dist
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

    pbar = tqdm.tqdm(total=total)

    while total:

        to_points = np.array(adj_edges[cur_point])
        cur_point = get_next_point(to_points, image, cur_point, used_hashed_points)

        total = total - 1
        used_hashed_points[cur_point] = 1
        path_hashed_points.append(cur_point)

        pbar.update(1)

    pbar.close()

    return path_hashed_points, adj_edges


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


import networkx as nx


def _create_eulerian_path(eulerianGraph: nx.MultiGraph, start: int):
    """
    オイラーグラフからオイラー路を生成する

    Parameters
    ----------
    eulerianGraph : networkx.MultiGraph
        オイラーグラフ
    start : int
        オイラー路のスタート地点

    Returns
    -------
    eulerianPath : list
        オイラー路を辿る頂点の順番のリスト
    """

    # オイラー路の辺リストを生成
    eulerianEdges = list(nx.eulerian_circuit(eulerianGraph, start))

    # オイラー路を辿る頂点の順番のリストを生成
    eulerianPath = [edge[0] for edge in eulerianEdges]
    # オイラー路のスタート地点とゴール地点を一致させる
    eulerianPath.append(eulerianEdges[len(eulerianEdges) - 1][1])

    return eulerianPath


def _create_hamiltonian_path(eulerianPath: list):
    """
    オイラー路からハミルトン閉路を生成する

    Parameters
    ----------
    eulerian_path : list
        オイラー路を辿る頂点の順番のリスト

    Returns
    -------
    hamiltonianPath : list
        ハミルトン閉路を辿る頂点の順番のリスト
    """

    # ハミルトン閉路を辿る頂点の順番のリストを初期化
    hamiltonianPath = []
    # 既出の頂点集合を初期化
    existedVertice = set()
    # オイラー路を辿る各頂点を辿り、2回目以降に現れた頂点は無視する
    for vertex in eulerianPath:
        if vertex not in existedVertice:
            hamiltonianPath.append(vertex)
            existedVertice.add(vertex)

    # ハミルトン閉路のスタート地点とゴール地点を一致させる
    hamiltonianPath.append(eulerianPath[0])

    return hamiltonianPath


def christofides_algorithm(costMatrix: list, start: int):
    """
    Christofidesのアルゴリズムで近似巡回ルートを生成する

    Parameters
    ----------
    costMatrix : list
        完全グラフのコスト行列
    start : int
        近似巡回ルートのスタート地点

    Returns
    -------
    route : list
        近似巡回ルート
    """

    # 1. コスト行列から重み付き完全グラフを生成
    graph = _create_weighted_graph(costMatrix)
    # 2. Primのアルゴリズムで最小全域木を生成
    spanningTree = nx.minimum_spanning_tree(graph, algorithm="prim")
    # 3. 最小全域木から偶数次数の頂点を除去
    removedGraph = _remove_even_degree_vertices(graph, spanningTree)
    # 4. 除去された最小全域木から最小コストの完全マッチングによるグラフを生成
    matchingGraph = _match_minimal_weight(removedGraph)
    # 5. 最小全域木と完全マッチングによるグラフを合体
    mergedGraph = _merge_two_graphs(spanningTree, matchingGraph)
    # 6. 合体したグラフからオイラー路を生成
    eulerianPath = _create_eulerian_path(mergedGraph, start)
    # 7. オイラー路からハミルトン閉路を生成
    route = _create_hamiltonian_path(eulerianPath)
    # 8. ハミルトン閉路を出力して終了
    return route


def _remove_even_degree_vertices(graph: nx.Graph, spanningTree: nx.Graph):
    """
    全域木から偶数次数の頂点をグラフから取り除いた部分ブラフを生成する

    Parameters
    ----------
    graph : networkx.Graph
        グラフ
    spanningTree : networkx.Graph
        全域木

    Returns
    -------
    removedGraph : networkx.Graph
       頂点を取り除いた部分グラフ
    """

    # 引数のグラフからコピーして初期化し、全域木の偶数次数の頂点を削除
    removedGraph = nx.Graph(graph)
    for v in spanningTree:
        if spanningTree.degree[v] % 2 == 0:
            removedGraph.remove_node(v)

    return removedGraph


def _match_minimal_weight(graph: nx.Graph):
    """
    グラフの最小コストの完全マッチングを生成する

    Parameters
    ----------
    graph : networkx.Graph
        グラフ

    Returns
    -------
    matchingGraph : set
        マッチングを構成する辺(2要素のみ持つタプル)のみ持つグラフ
    """

    # グラフの全コストの大小関係を反転させるため、コストの符号を逆にして初期化
    tmpGraph = nx.Graph()
    for edgeData in graph.edges.data():
        tmpGraph.add_edge(edgeData[0], edgeData[1], weight=-edgeData[2]["weight"])
    # ブロッサムアルゴリズムで最大重み最大マッチングを生成
    matching = nx.max_weight_matching(tmpGraph, maxcardinality=True)

    # マッチングを構成するのみ持つグラフの生成
    matchingGraph = nx.Graph()
    for edge in matching:
        matchingGraph.add_edge(
            edge[0], edge[1], weight=graph[edge[0]][edge[1]]["weight"]
        )

    return matchingGraph


def _merge_two_graphs(graph1: nx.Graph, graph2: nx.Graph):
    """
    辺が2重化されていない2つのグラフを合体する

    Parameters
    ----------
    graph1 : networkx.Graph
        1つ目のグラフ
    graph2 : networkx.Graph
        2つ目のグラフ

    Returns
    -------
    mergedGraph : networkx.MultiGraph
        合体したグラフ
    """

    # 合体したグラフを1つ目のグラフからコピーして初期化し、2つ目のグラフの各辺を追加
    mergedGraph = nx.MultiGraph(graph1)
    for edgeData in graph2.edges.data():
        mergedGraph.add_edge(edgeData[0], edgeData[1], weight=edgeData[2]["weight"])

    return mergedGraph


def main():
    df_image = pd.read_csv("../input/santa-2022/image.csv")
    image = df_to_image(df_image)

    mat = generate_adj_mat(image)
    # mst, edges = get_mst_edges(mat)
    # plot_mst(edges, image)

    graph = nx.from_scipy_sparse_matrix(mat, create_using=nx.MultiGraph)

    spanningTree = nx.minimum_spanning_tree(graph, algorithm="prim")
    print(1)
    # 3. 最小全域木から偶数次数の頂点を除去
    removedGraph = _remove_even_degree_vertices(graph, spanningTree)
    print(2)
    # 4. 除去された最小全域木から最小コストの完全マッチングによるグラフを生成
    matchingGraph = _match_minimal_weight(removedGraph)
    print(3)
    # 5. 最小全域木と完全マッチングによるグラフを合体
    mergedGraph = _merge_two_graphs(spanningTree, matchingGraph)
    print(4)
    # 6. 合体したグラフからオイラー路を生成
    eulerianPath = _create_eulerian_path(mergedGraph, xy_to_hash(radius, radius))
    print(5)
    # 7. オイラー路からハミルトン閉路を生成
    route = _create_hamiltonian_path(eulerianPath)
    print(6)

    with open("route.pickle", mode="wb") as f:
        pickle.dump(route, f)
