import numpy as np

def norm_entropy(arr: [np.ndarray]):
    """
    Calculates the normalized entropy of a given array. The array must contain all zeros, if necessary.
    The normalized entropy is defined as the entropy of the array divided by the maximum entropy, log2(len(array)).
    """
    if np.sum(arr) == 0:
        return 1
    else:
        arr_nonzero = arr[arr > 0]
        return - arr_nonzero * np.log2(arr_nonzero) / np.log2(len(arr))