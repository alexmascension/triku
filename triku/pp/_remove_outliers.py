import numpy as np
import scipy.sparse as spr


def remove_outliers(arr: np.ndarray, sigma: float = 5.0, do_copy: bool = False,):
    if do_copy:
        arr = arr.copy()

    arr = arr.astype(float)
    arr[arr == 0] = np.NaN
    mean, std = np.asarray(np.nanmean(arr, axis=0)).flatten(), np.asarray(np.nanstd(arr, axis=0)).flatten()
    np.nan_to_num(arr, 0)
    arr[arr > sigma * std] = np.mean(arr)  # This is not the best, but it is sufficiently fast

    if do_copy:
        return arr
