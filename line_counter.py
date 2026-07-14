CROSSING_NONE = 0
CROSSING_IN = 1
CROSSING_OUT = 2


def cross_product(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def which_side(line, point):
    x1, y1, x2, y2 = line
    cp = cross_product(x1, y1, x2, y2, point[0], point[1])
    return 1 if cp >= 0 else -1


def detect_crossing(line, old_centroid, new_centroid, flip=False):
    if line is None or old_centroid is None or new_centroid is None:
        return CROSSING_NONE
    old_side = which_side(line, old_centroid)
    new_side = which_side(line, new_centroid)
    if old_side != new_side:
        if flip:
            return CROSSING_OUT if new_side == 1 else CROSSING_IN
        else:
            return CROSSING_IN if new_side == 1 else CROSSING_OUT
    return CROSSING_NONE
