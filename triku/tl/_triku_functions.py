import numpy as np
import scipy.stats as sts
import scipy.sparse as spr
from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from umap.umap_ import nearest_neighbors

import ray

from triku.utils import return_proportion_zeros, return_mean, check_count_mat, find_starting_point, distance
from triku.logg import triku_logger


def return_knn_indices(array: np.ndarray, knn: int, return_random: bool, random_state: int, metric: str):
    """
    Given a expression array and a number of kNN, returns a n_cells x kNN + 1 matrix where each row, col is a
    neighbour of cell X.

    return_random attribute is used to assign random neighbours.
    """

    pca = PCA(n_components=50, whiten=True, svd_solver='auto').fit_transform(array)

    if return_random:
        knn_indices = np.zeros((array.shape[0], knn))
        ran = np.arange(0, array.shape[0]) # TODO: remove for loop and instead use np.random.randint(array.shape[0], array.size) and reshape
        for i in ran:  # TODO: revise if kNN matrix yields kNN + 1 or kNN neighbours
            knn_indices[i, 1:] = np.random.choice(ran, knn - 1, replace=False)

        knn_indices[:, 0] = np.arange(array.shape[0])

    else:
        knn_indices, knn_dists, forest = nearest_neighbors(pca, n_neighbors=knn, metric=metric,
                                                           random_state=np.random.RandomState(random_state),
                                                           angular=False, metric_kwds={})

    return knn_indices.astype(int)


def return_knn_expression(arr_expression: np.ndarray, knn_indices: np.ndarray):
    """
    This function returns a dictionary of genes: expression in kNN for gene. To calculate the expression per gene
    we are going to apply the following procedure.

    First we create a 2D mask of neighbors. The mask is a translation of the knn_indices into a 2D sparse array,
    where the index i,j is 1 if cell j is neigbour of cell i and 0 elsewhere.

    That is, if we have n_g as number of genes, and n_c as number of cells, the matrix product would be:

        Mask          Expr         Result
    (n_c x n_c) · (n_c x n_g) = (n_c x n_g)

    Then, the Result matrix would have in each cell, the summed expression of that gene in the knn (and also the
    own cell).
    """

    sparse_mask = spr.lil_matrix((arr_expression.shape[0], arr_expression.shape[0]))
    # [:, 0] is [0,0,0,0,..., 0, 1, ..., 1, ... ] and [:, 1] are the indices of the rest of cells.
    sparse_mask[np.repeat(np.arange(knn_indices.shape[0]), knn_indices.shape[1]), knn_indices.flatten()] = 1

    return sparse_mask.dot(arr_expression)


def create_random_count_matrix(matrix: np.array = None):
    """
    Given a matrix with cells x genes, returns a randomized cells x genes matrix. This matrix has, for each genes,
    the counts of the gene from the original matrix dispersed across the cells. E.g., if gene X has 1000 across
    all cells counts, those counts are distributed randomly.
    """

    n_reads_per_gene = matrix.sum(0).astype(int)
    print(n_reads_per_gene)
    n_cells, n_genes = matrix.shape
    matrix_random = np.zeros((n_genes, n_cells))

    # The limiting part generally is the random number generation.
    # Random.choice is rather slow, so to save some time we use random.random, then multiply by the
    # number of cells, and change to int.
    random_counts = np.random.randint(n_cells, size=np.sum(n_reads_per_gene))

    # Also, assigning values to a matrix is done by rows because it is 2 to 3 times faster than in rows.
    # Numpy rows are row-based so it will always be more efficient to do a row-wise assignment.
    idx_counts = 0
    for gene in range(n_genes):
        counts_gene = random_counts[idx_counts: idx_counts + n_reads_per_gene[gene]]
        bincount = np.bincount(counts_gene, minlength=n_cells)
        matrix_random[gene, :] = bincount
        idx_counts += n_reads_per_gene[gene]

    matrix_random = matrix_random.T
    return matrix_random


# TODO: apply for log-transformed data? The convolution works assuming that X data are integers.
def apply_convolution_read_counts(probs: np.ndarray, knn: int):
    """
    Convolution of functions. The function applies a convolution using np.convolve
    of a probability distribution knn times. The result is an array of N elements (N arises as the convolution
    of a n-length array knn times) where the element i has the probability of i being observed.

    Parameters
    ----------
    probs : np.array
        Object with count matrix. If `pandas.DataFrame`, rows are cells and columns are genes.
    knn : int
        Number of kNN
    """
    # We are calculating the convolution of cells with positive expression. Thus, in the first distribution
    # we have to remove the cells with 0 reads, and rescale the probabilities.
    arr_0 = probs.copy()
    arr_0[0] = 0  # TODO: this will fail in log-transformed data
    arr_0 /= arr_0.sum()

    # We will use arr_bvase as the array with the read distribution
    arr_base = probs.copy()

    arr_convolve = np.convolve(arr_0, arr_base, )

    for knni in range(2, knn):
        arr_convolve = np.convolve(arr_convolve, arr_base, )

    # TODO: check the probability sum is 1 and, if so, remove
    arr_prob = arr_convolve / arr_convolve.sum()

    # TODO: if log transformed, this is untrue. Should not be arange.
    return np.arange(len(arr_prob)), arr_prob


def compute_conv_idx(counts_gene, knn):
    """
    Given a GENE x CELL matrix, and an index to select from, calculates the convolution of reads for that gene index.
    The function returns the
    """
    x_counts, y_counts = np.unique(counts_gene, return_counts=True)
    y_probs = y_counts / y_counts.sum()  # Important to transform count to probabilities
    # to keep the convolution constant.

    x_conv, y_conv = apply_convolution_read_counts(y_probs, knn=knn)

    return x_conv, y_conv, y_probs


def calculate_emd(knn_counts, x_conv, y_conv):
    """
    Returns "normalized" earth movers distance (EMD). The function calculates the x positions and probabilities
    of the "real" dataset using the knn_counts, and the x positions and probabilities of the convolution as attributes.

    To normalize the distance, it is divided by the standard deviation of the convolution. Since the convolution
    is already given as a distribution, mean and variance have to be calculated "by hand".
    """
    dist_range = np.arange(max(knn_counts) + 1)
    # np.bincount transforms [3, 3, 4, 1, 2, 9] into [0, 1, 1, 2, 1, 0, 0, 0, 0, 1]
    real_vals = np.bincount(knn_counts.astype(int)) / len(knn_counts)

    emd = sts.wasserstein_distance(dist_range, x_conv, real_vals, y_conv)

    mean = (x_conv * y_conv).sum()
    std = np.sqrt(np.sum(y_conv * (x_conv - mean) ** 2))

    return emd / std


@ray.remote
def compute_convolution_and_emd(array_counts, array_knn_counts, idx, knn,):
    counts_gene = array_counts[idx, :].ravel()  # idx is chosen by rows, because it is more effective!
    knn_counts = array_knn_counts[idx, :].ravel()

    x_conv, y_conv, y_probs = compute_conv_idx(counts_gene, knn)
    emd = calculate_emd(knn_counts, x_conv, y_conv)

    return x_conv, y_conv, emd

# TODO comentar
def parallel_emd_calculation(array_counts: np.ndarray, array_knn_counts: np.ndarray, n_procs: int, knn: int):

    # TODO: possible memory leak! assert if transposition is applied after return and, if so, revert at the end of func
    array_counts = array_counts.copy().T  # to make genes be rows, the indexing is faster!
    array_knn_counts = array_knn_counts.copy().T

    ray.init(num_cpus=n_procs, ignore_reinit_error=True)

    array_counts_id = ray.put(array_counts)
    array_knn_counts_id = ray.put(array_knn_counts)

    ray_obj_ids = [compute_convolution_and_emd.remote(array_counts_id, array_knn_counts_id, idx_gene, knn)
                   for idx_gene in range(array_counts.shape[0])]
    ray_objs = ray.get(ray_obj_ids)

    list_x_conv, list_y_conv, list_emd = [x[0] for x in ray_objs], [x[1] for x in ray_objs], [x[2] for x in ray_objs]

    return list_x_conv, list_y_conv, np.array(list_emd)



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
        triku_logger.warning("Mean difference for this bin is zero (n_0 = {}, n_f = {}, mean = {}). "
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

    triku_logger.info("Binning the dataset")
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

    triku_logger.info("Getting selected genes")
    # Remove duplicated entries
    selected_genes_index = sorted(list(dict.fromkeys(selected_genes_index)))

    return selected_genes_index
