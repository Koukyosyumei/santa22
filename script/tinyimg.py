import cv2
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

offset = 1
color_weight = 0.3  # 0.3
INF = 1e6


def _create_weighted_graph(costMatrix: list):
    """
    完全グラフの合計コストで重み付けした完全グラフを生成する

    Parameters
    ----------
    costMatrix : list
        完全グラフのコスト行列

    Returns
    -------
    graph : networkx.Graph
        重み付き完全グラフ
    """

    # 重み付き完全グラフを初期化して辺を追加
    graph = nx.Graph()
    for i in range(len(costMatrix)):
        for j in range(i + 1, len(costMatrix[i])):
            if costMatrix[i][j] != INF:
                graph.add_edge(i, j, weight=costMatrix[i][j])

    return graph


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
    print("Success: Initial graph generation")
    # 2. Primのアルゴリズムで最小全域木を生成
    spanningTree = nx.minimum_spanning_tree(graph, algorithm="prim")
    print("Success: MCT generation with prim")
    # 3. 最小全域木から偶数次数の頂点を除去
    removedGraph = _remove_even_degree_vertices(graph, spanningTree)
    print("Success: Remove vertices with even degree")
    # 4. 除去された最小全域木から最小コストの完全マッチングによるグラフを生成
    matchingGraph = _match_minimal_weight(removedGraph)
    print("Success: Graph generation with min-cost matching")
    # 5. 最小全域木と完全マッチングによるグラフを合体
    mergedGraph = _merge_two_graphs(spanningTree, matchingGraph)
    print("Success: Graph matching")
    # 6. 合体したグラフからオイラー路を生成
    eulerianPath = _create_eulerian_path(mergedGraph, start)
    print("Success: Eulerian generation")
    # 7. オイラー路からハミルトン閉路を生成
    route = _create_hamiltonian_path(eulerianPath)
    print("Success: Hamiltonian generation")
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


def _duplicate_weighted_graph(graph: nx.Graph):
    """
    重み付けグラフの辺を2重化する

    Parameters
    ----------
    graph : networkx.Graph
        重み付けグラフ

    Returns
    -------
    duplicatedGraph : networkx.MultiGraph
        引数の重み付けグラフの辺を2重化した重み付けグラフ
    """

    # 重み付けグラフを初期化して引数のグラフの各辺を2重化
    duplicatedGraph = nx.MultiGraph()
    for v in graph:
        for w in graph[v]:
            duplicatedGraph.add_edge(v, w, weight=graph[v][w]["weight"])

    return duplicatedGraph


def double_tree_algorithm(costMatrix: list, start: int):
    """
    2重木アルゴリズムで近似巡回ルートを生成する

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
    print(1)
    # 2. Primのアルゴリズムで最小全域木を生成
    spanningTree = nx.minimum_spanning_tree(graph, algorithm="prim")
    print(2)
    # 3. 最小全域木の各辺を2重化
    duplicatedSpanningTree = _duplicate_weighted_graph(spanningTree)
    print(3)
    # 4. 2重化した最小全域木からオイラー路を生成
    eulerianPath = _create_eulerian_path(duplicatedSpanningTree, start)
    print(4)
    # 5. オイラー路からハミルトン閉路を生成
    route = _create_hamiltonian_path(eulerianPath)
    print(5)
    # 6. ハミルトン閉路を出力して終了
    return route


def plot_traj(points, image_, filename_):
    lines = []
    for i_ in range(1, len(points)):
        lines.append([points[i_ - 1], points[i_]])

    colors = []
    for l in lines:
        dist = np.abs(l[0] - l[1]).max()
        if dist <= 2:
            colors.append("b")
        else:
            colors.append("r")

    lc = mc.LineCollection(lines, colors=colors)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.add_collection(lc)

    ax.matshow(image_)
    ax.grid(None)

    ax.autoscale()
    fig.savefig(filename_)


def main():
    img = cv2.imread("data/img_tiny.png") / 255
    print(img.max())
    num_points = img.shape[0] * img.shape[1]
    cost_matrix = np.ones((num_points, num_points)) * INF

    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            pos = i + j * img.shape[1]
            for i_ in range(max(0, i - offset), min(i + offset + 1, img.shape[0])):
                for j_ in range(max(0, j - offset), min(j + offset + 1, img.shape[1])):
                    assert abs(i - i_) in [0, 1]
                    assert abs(j - j_) in [0, 1]
                    if i != i_ or j != j_:
                        cost_matrix[pos, i_ + j_ * img.shape[1]] = np.sqrt(
                            (i - i_) ** 2 + (j - j_) ** 2
                        ) + color_weight * np.sum(np.abs(img[i, j] - img[i_, j_]))

    route = double_tree_algorithm(cost_matrix, 32 * 64 + 32)
    points = [np.array(((p // 64), p % 64)) for p in route]

    plot_traj(points, img, "a.png")
    plot_traj(points[:100], img, "b.png")
    plot_traj(points[:1000], img, "c.png")
    plot_traj(points[3000:], img, "d.png")

    pd.DataFrame(np.array(points)).to_csv("doule_tree_64x64.csv", index=False)


if __name__ == "__main__":
    main()
