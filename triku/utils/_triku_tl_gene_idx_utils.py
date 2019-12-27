import numpy as np

def find_starting_point(x, y, delta_y=None, delta_x=None):
    delta_y = (max(y) - min(y)) / 20 if delta_y is None else delta_y
    delta_x = int(len(x) / 7) if delta_x is None else delta_x

    for x_stop in range(len(x), int(len(x) / 2), -1):
        y_box = y[x_stop - delta_x: x_stop]
        y_diff = max(y_box) - min(y_box)
        if y_diff < delta_y:
            return x_stop - delta_x

    return int(len(x) / 2)


def distance(m, b, x, y):
    if np.isinf(m):
        return 0
    return (y - m * x - b) / ((1 + m ** 2) ** 0.5)
