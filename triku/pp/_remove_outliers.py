import numpy as np
import scipy.sparse as spr
from time import time
import bottleneck as bn
from numba import njit


def remove_outliers(arr: np.ndarray, sigma: float = 5.0, do_copy: bool = False,):
    if do_copy:
        arr = arr.copy()

    t = time()
    arr = arr.astype(float)
    arr[arr == 0] = np.NaN
    std, median = bn.nanstd(arr, axis=0).flatten(), bn.nanmedian(arr, axis=0).flatten()
    np.nan_to_num(arr, 0)
    means_array = np.repeat(median.reshape(1, arr.shape[1]), arr.shape[0], axis=0)
    mask = arr > sigma * std
    arr[mask], means_array[~mask] = 0, 0
    arr = arr + means_array

    if do_copy:
        return arr
