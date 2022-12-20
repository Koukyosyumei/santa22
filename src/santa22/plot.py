import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np


def plot_traj(points, image_):
    origin = np.array([0, 0])
    lines = []
    if not (origin == points[0]).all():
        lines.append([origin, points[0]])
    for i_ in range(1, len(points)):
        lines.append([points[i_ - 1], points[i_]])
    if not (origin == points[1]).all():
        lines.append([points[-1], origin])

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

    radius = image_.shape[0] // 2
    ax.matshow(
        image_ * 0.8 + 0.2,
        extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5),
    )
    ax.grid(None)

    ax.autoscale()
    fig.show()
