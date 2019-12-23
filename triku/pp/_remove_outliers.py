import numpy as np


def remove_outliers(arr: np.ndarray, sigma: float = 5.0, do_copy: bool = False, ):
    if do_copy:
        arr = arr.copy()

    for gene in range(arr.shape[1]):
        expression_list = arr[:, gene]
        expresion_positive = expression_list[expression_list > 0]
        mean, std = np.mean(expresion_positive), np.std(expresion_positive)
        idx = np.argwhere(expression_list > sigma * std).flatten()
        arr[idx, gene] = mean

    if do_copy:
        return arr
