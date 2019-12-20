import numpy as np
from scipy.signal import savgol_filter

from triku.utils import return_proportion_zeros, return_mean, check_count_mat
from triku.logg import logger


def find_starting_point(x, y, delta_y=None, delta_x=None):
    delta_y = (max(y) - min(y)) / 20 if delta_y is None else delta_y
    delta_x = int(len(x) / 7) if delta_x is None else delta_x

    for x_stop in range(len(x), int(len(x) / 2), -1):
        y_box = y[x_stop - delta_x: x_stop]
        y_diff = max(y_box) - min(y_box)
        if y_diff < delta_y:
            return x_stop

    return int(len(x) / 2)


def distance(m, b, x, y):
    if np.isinf(m):
        return 0
    return y - m * x - b / ((1 + m ** 2) ** 0.5)


def find_knee_point(x, y, s=0.0):
    """
    Calculates the knee point of a curve. The knee point here is defined as the
    furthest distance between the curve C, and the line that joins the two extremes of the curve, R.
    This knee point, although not the most robust, is easy to calculate and interpret.

    The shortest distance between a c in C and R is (y_c - (m_r * x_c + b_r)) / (sqrt(1 + m_r ^ 2)). The knee point
    is, therefore, the largest of the distances.
    """
    x_0, x_f, y_0, y_f = x[0], x[-1], y[0], y[-1]
    m = (y_f - y_0) / (x_f - x_0)
    b = y_0 - m * x_0

    list_d = [distance(m, b, x[i], y[i]) for i in range(len(x))]

    if len(list_d) < 3:
        return 0, False

    if s >= 0:
        correction_factor = min(s, 0.9)
        knee_x_idx = np.argwhere(list_d <= (1 - correction_factor) * min(list_d)).flatten()[-1]
    else:
        correction_factor = min(abs(s), 0.9)
        knee_x_idx = np.argwhere(list_d <= (1 - correction_factor) * min(list_d)).flatten()[0]

    return knee_x_idx, True


def return_idx(prop_0: np.ndarray, mean: np.ndarray, percentile_low: float, percentile_high: float, s: float):
    """
    Selection of the knee point in a curve. This function accepts the list of zero proportions, means, the
    cutting points, and the s parameter, and returns the indices from prop_0 whose mean is greater than the
    one at the knee point. How the curve is obtained and how s acts, is explained in `find_knee_point`.
    """

    selector = (prop_0 >= percentile_low) & (prop_0 <= percentile_high)
    array_mean = mean[selector]

    # Calculate the curve of number of genes by mean. This curve is scaled from 0 to 1 in both axes.
    if len(array_mean) == 0:
        return []

    x = np.arange(len(array_mean)) / len(array_mean)
    y = (np.sort(np.log10(array_mean)) - min(np.log10(array_mean))) / \
        (np.max(np.log10(array_mean))  - np.min(np.log10(array_mean)))

    # Apply a Savitzky-Golay filter to remove noisiness on the curve. This filter will be applicable to arrays
    # longer than 21 elements. This number is more or less arbitrary, but we have seen that for curves with fewer
    # points it does introduce some artifacts.
    if len(y) > 21:
        y = savgol_filter(y, 2 * int(len(x) / 15) + 1, 3)
    x_stop = find_starting_point(x, y)
    knee_x_idx, status = find_knee_point(x[x_stop:], y[x_stop:], s)

    if not status:  # There might not be a knee point because of the curve, or for any other reason
        return []

    knee_x_idx += x_stop
    knee_mean = np.sort(array_mean)[knee_x_idx]

    selector_idxs = np.argwhere((prop_0 > percentile_low) & (prop_0 < percentile_high) &
                                (mean > knee_mean)).flatten().tolist()

    return selector_idxs


def return_triku_gene_idx(arr: np.ndarray, n_bins: int = 80, n_cycles: int = 4, s: float = 0):
    """
    Returns the indices of the array corresponding to the selected genes, with high variability.
    Parameters are explained in the main triku function.
    """

    # Prepare the matrix, get the mean and the proportion of zeros, and resolve other attributes
    check_count_mat(arr)
    mean, prop_0 = return_mean(arr), return_proportion_zeros(arr)

    """
    In previous versions we removed top and bottom percentage genes because they seemed not to be relevant. Now 
    we will remove them a posteriori applying an entropy threshold.
    """

    # Divide the dataset into bins. Each bin should have a similar number of genes to analyze. Thus, ranges with
    # fewer number of genes will be larger, and ranges with further number of genes will be smaller.
    prop_0_bins = [np.percentile(prop_0, 100 * i / n_bins) for i in range(n_bins)]

    # For each bin, we will take the genes that pass the threshold. This threshold is calculated as the knee point
    # of the rank of mean VS mean. To make the threshold across the whole prop_0 range, we will shift the range of
    # 0_prop by a small amount. This amount will be equivalent to len_bin / n_cycles. Thus, we will better cover
    # the whole interval, and not make the gene selection stepper.
    selected_genes_index = []

    for N in range(n_bins - 1):
        len_bin = prop_0_bins[N + 1] - prop_0_bins[N]

        for cycle in range(n_cycles):
            start_prop = prop_0_bins[N] + cycle * len_bin / n_cycles
            end_prop = prop_0_bins[N + 1] + cycle * len_bin / n_cycles
            selector_idxs = return_idx(prop_0, mean, start_prop, end_prop, s)

            selected_genes_index += selector_idxs

    # Remove duplicated entries
    selected_genes_index = sorted(list(dict.fromkeys(selected_genes_index)))

    return selected_genes_index
