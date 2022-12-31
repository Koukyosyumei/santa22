import pandas as pd

from santa22.mst import (
    df_to_image,
    generate_adj_mat,
    get_hashed_points_order,
    get_mst_edges,
    plot_mst,
)


def main():
    df_image = pd.read_csv("data/image.csv")
    image = df_to_image(df_image)
    mat = generate_adj_mat(image)
    edges = get_mst_edges(mat)
    path_hashed_points = get_hashed_points_order(edges, image)

    print(len(path_hashed_points))
    plot_mst(edges, image)


if __name__ == "__main__":
    main()
