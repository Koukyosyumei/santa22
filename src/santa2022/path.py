import numpy as np

from .action import get_direction, get_position, get_radii, get_radius, rotate


def get_path_to_point(config, point_):
    """Find a path of configurations to `point` starting at `config`."""
    config_start = config.copy()
    radii = get_radii(config)

    # Rotate each link, starting with the largest, until the point can
    # be reached by the remaining links. The last link must reach the
    # point itself.
    for i_ in range(len(config)):
        link = config[i_]
        base = get_position(config[:i_])
        relbase = point_ - base
        position = get_position(config[: i_ + 1])
        relpos = point_ - position
        radius = radii[i_ + 1]

        # Special case when next-to-last link lands on point.
        if radius == 1 and (relpos == 0).all():
            config = rotate(config, i_, 1)
            if (get_position(config) == point_).all():
                break
            else:
                continue
        while np.max(np.abs(relpos)) > radius:
            direction = get_direction(link, relbase)
            config = rotate(config, i_, direction)
            link = config[i_]
            base = get_position(config[:i_])
            relbase = point_ - base
            position = get_position(config[: i_ + 1])
            relpos = point_ - position
            radius = get_radius(config[i_ + 1 :])

    assert (get_position(config) == point_).all()
    path = get_path_to_configuration(config_start, config)

    return path


def get_path_to_configuration(from_config, to_config):
    path = np.expand_dims(from_config, 0).copy()
    config = from_config.copy()
    while (config != to_config).any():
        for i_ in range(len(config)):
            config = rotate(config, i_, get_direction(config[i_], to_config[i_]))
        path = np.append(path, np.expand_dims(config, 0), 0)
    assert (path[-1] == to_config).all()
    return path


def get_origin(size):
    assert size % 2 == 1
    radius = size // 2
    p = [1]
    for power in range(0, 8):
        p.append(2**power)
        if sum(p) == radius:
            break
    else:
        assert False
    p = p[::-1]
    config = np.array([(p[0], 0)] + [(-pp, 0) for pp in p[1:]])
    return config


def points_to_path(points, size=257):
    origin = get_origin(size)
    visited = set()
    path = [origin]
    for p in points:
        config = path[-1]
        if tuple(p) not in visited:
            candy_cane_road = get_path_to_point(config, p)[1:]
            if len(candy_cane_road) > 0:
                visited |= set([tuple(get_position(r)) for r in candy_cane_road])
            path.extend(candy_cane_road)
    # Back to origin
    candy_cane_road = get_path_to_configuration(path[-1], origin)[1:]
    visited |= set([tuple(get_position(r)) for r in candy_cane_road])
    path.extend(candy_cane_road)

    assert (
        len(visited) == size**2
    ), f"Visited {len(visited)} points out of {size ** 2}"

    return np.array(path)
