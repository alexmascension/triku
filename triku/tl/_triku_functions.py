import numpy as np
from scipy.signal import savgol_filter

from triku.utils import return_proportion_zeros, return_mean, check_count_mat, find_starting_point, distance
from triku.logg import logger



def find_knee_point(x, y, s=0.0):
    """
    Calculates the knee point of a curve. The knee point here is defined as the
    furthest distance between the curve C, and the line that joins the two extremes of the curve, R.
    This knee point, although not the most robust, is easy to calculate and interpret.

    The shortest distance between a c in C and R is (y_c - (m_r * x_c + b_r)) / (sqrt(1 + m_r ^ 2)). The knee point
    is, therefore, the largest of the distances. To obtain a more robust knee point, we will get first an intermediate
    point which has a small slope.

    After that we correct the knee_point selection using the factor s. s is a factor that decreases the maximum distance
    described before, selecting a point further left or right in the curve. A positive s will select a knee point
    further right (higher mean - fewer points will be selected), and a negative s will select a knee point further left
    (lower mean - more points will be selected).
    """
    x_0, x_f, y_0, y_f = x[0], x[-1], y[0], y[-1]
    m = (y_f - y_0) / (x_f - x_0)
    b = y_0 - m * x_0

    list_d = [distance(m, b, x[i], y[i]) for i in range(len(x))]

    if len(list_d) < 3:
        return 0

    s = -s  # We change that so that s now follows intuitively. Greater s implies more genes and vice-versa.

    if s >= 0:
        correction_factor = min(s, 0.9)
        knee_x_idx = np.argwhere(list_d <= (1 - correction_factor) * min(list_d)).flatten()[-1]
    else:
        correction_factor = min(abs(s), 0.9)
        knee_x_idx = np.argwhere(list_d <= (1 - correction_factor) * min(list_d)).flatten()[0]

    return knee_x_idx


def return_idx(prop_0: np.ndarray, mean: np.ndarray, percentile_low_idx: int, percentile_high_idx: int, s: float,
               delta_x: int = None, delta_y: int = None, apply_deltas: bool = True):
    """
    Selection of the knee point in a curve. This function accepts the list of zero proportions, means, the
    cutting points, and the s parameter, and returns the indices from prop_0 whose mean is greater than the
    one at the knee point. How the curve is obtained and how s acts, is explained in `find_knee_point`.
    """
    percentile_low, percentile_high = np.sort(prop_0)[percentile_low_idx], np.sort(prop_0)[percentile_high_idx],

    selector = (prop_0 >= percentile_low) & (prop_0 <= percentile_high)
    array_mean = mean[selector]

    # Calculate the curve of number of genes by mean. This curve is scaled from 0 to 1 in both axes.
    if np.min(array_mean) == np.max(array_mean):
        logger.warning("Mean difference for this bin is zero (n_0 = {}, n_f = {}, mean = {}). "
                       "This might not be a problem, but sometimes it is associated to poor data quality".format(
            percentile_low_idx, percentile_high_idx, np.min(array_mean)))
        return []

    x = np.arange(len(array_mean)) / len(array_mean)
    y = (np.sort(np.log10(array_mean)) - min(np.log10(array_mean))) / \
        (np.max(np.log10(array_mean)) - np.min(np.log10(array_mean)))


    '''
    Apply a Savitzky-Golay filter to remove noisiness on the curve. This filter will be applicable to arrays
    longer than 21 elements. This number is more or less arbitrary, but we have seen that for curves with fewer
    points it does introduce some artifacts.
    '''
    if len(x) > 36:  # 3 * 15 + 1
        y = savgol_filter(y, 2 * int(len(x) / 15) + 1, 3)
    '''
    In some cases the curve does a small step, or falls and rises at the beginning, and these two cases destabilize
    finding the knee point. To solve that problem we will "remove" those points from the curve, and then find the knee
    point using the rest of the curve. 
    
    We find that point with the function `find_starting_point`. This function is a sliding window from right (where
    the knee point is) to left, and, given a delta_x, creates a box of with delta_x and height y_f - y_0. If y_f - y_0
    is smaller than delta_y, then it returns the left point of the box.
    '''
    if apply_deltas:
        x_stop = find_starting_point(x, y, delta_x=delta_x, delta_y=delta_y)
    else:
        x_stop = 0

    knee_x_idx = find_knee_point(x[x_stop:], y[x_stop:], s)

    knee_x_idx += x_stop
    knee_mean = np.sort(array_mean)[knee_x_idx]

    selector_idxs = np.argwhere((prop_0 > percentile_low) & (prop_0 < percentile_high) &
                                (mean > knee_mean)).flatten().tolist()

    return selector_idxs


def return_triku_gene_idx(arr: np.ndarray, n_bins: int = 80, n_cycles: int = 4, s: float = 0, delta_x: int = None,
                          delta_y: int = None, seed=0):
    """
    Returns the indices of the array corresponding to the selected genes, with high variability.
    Parameters are explained in the main triku function.
    """

    # Prepare the matrix, get the mean and the proportion of zeros, and resolve other attributes
    mean, prop_0 = return_mean(arr), return_proportion_zeros(arr)

    # First we introduce a slight amount of noise into the proportions. This makes the binning more robust to
    # datasets that do not have smooth percentages. In those cases, the percentiles on the upper bound can fail,
    # and although it does not throw an error, it makes binning more unequal.
    np.random.seed(seed)
    prop_0 += np.random.uniform(low=0, high=(np.max(prop_0) - np.min(prop_0)) / (100 * n_bins), size=len(prop_0))

    """
    In previous versions we removed top and bottom percentage genes because they seemed not to be relevant. Now 
    we will remove them a posteriori applying an entropy threshold.
    """

    logger.info("Binning the dataset")
    # Divide the dataset into bins. THE PREVIOUS METHOD included the len_bin as prop_0_bins[N + 1] - prop_0_bins[N].
    # This is wrong because distribution of points are skewed towards points with high prop_0 / low mean, and the bins
    # for the low prop_0 / high mean are much larger, making the selection process skewed. To remove this we have to
    # make the length of the bin exactly the number of elements in the bin (which now is equal (+-1) due to random
    # noise added).

    # Each bin should have a similar number of genes to analyze. Thus, ranges with
    # fewer number of genes will be larger, and ranges with further number of genes will be smaller.
    prop_0_bins_idx = [int(i) for i in np.linspace(0, len(prop_0), n_bins + 1)]

    # For each bin, we will take the genes that pass the threshold. This threshold is calculated as the knee point
    # of the rank of mean VS mean. To make the threshold across the whole prop_0 range, we will shift the range of
    # 0_prop by a small amount. This amount will be equivalent to len_bin / n_cycles. Thus, we will better cover
    # the whole interval, and not make the gene selection stepper.
    selected_genes_index = []

    for N in range(n_bins - 1):
        len_bin = prop_0_bins_idx[N + 1] - prop_0_bins_idx[N]

        for cycle in range(n_cycles):
            start_prop_idx = prop_0_bins_idx[N] + int(cycle * len_bin / n_cycles)
            end_prop_idx = prop_0_bins_idx[N + 1] + int(cycle * len_bin / n_cycles)
            selector_idxs = return_idx(prop_0, mean=mean, s=s, delta_x=delta_x, delta_y=delta_y,
                                       percentile_low_idx=start_prop_idx, percentile_high_idx=end_prop_idx, )
            selected_genes_index += selector_idxs

    logger.info("Getting selected genes")
    # Remove duplicated entries
    selected_genes_index = sorted(list(dict.fromkeys(selected_genes_index)))

    return selected_genes_index


