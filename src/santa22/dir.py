from numba import jit


@jit
def dir2idx(dx, dy):
    assert -1 <= dx <= 1 and -1 <= dy <= 1
    assert not (dx == 0 and dy == 0)
    idx = (dx + 1) * 3 + dy + 1
    if idx > 4:
        idx = idx - 1
    return idx


@jit
def idx2dir(idx):
    assert 0 <= idx < 8
    if idx >= 4:
        idx = idx + 1
    dx = idx // 3 - 1
    dy = idx % 3 - 1
    return dx, dy


@jit
def idx_2_new_ij(i_, j_, idx):
    dx, dy = idx2dir(idx)
    new_i = i_ - dy
    new_j = j_ + dx
    return new_i, new_j
