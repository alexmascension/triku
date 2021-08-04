import gc
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as spr
import scipy.stats as sts
from scipy.signal import fftconvolve

from triku.logg import TRIKU_LEVEL
from triku.logg import triku_logger

# import logging
# from tqdm import tqdm
# from triku.genutils import TqdmToLogger


def clean_adata(adata):
    for varx in [
        "triku_distance",
        "triku_distance_uncorrected",
        "triku_distance_random",
    ]:
        if varx in adata.var:
            del adata.var[varx]

    if "triku_params" in adata.uns:
        del adata.uns["triku_params"]

    return adata


def save_object_triku(dict_triku, list_genes, path):
    df = pd.DataFrame(dict_triku)
    df = df.set_index(list_genes, drop=True)
    triku_logger.info(f"Saving triku return in {path}")
    df.to_csv(path, sep=",")


def return_knn_array(object_triku):
    # Distances array contains a pairwise relationship between cells, based on distance.
    # We will binarize that array to select equally all components with non-zero distance.
    # We finally add the identity matrix to select as neighbour the own cell.
    try:
        knn_array = (
            object_triku.obsp[  # type:ignore
                "distances"
            ]
            > 0
        ) + spr.identity(len(object_triku)).astype(
            bool
        )  # Saves memory

    except KeyError:
        triku_logger.warning(
            """Deprecation issue. Recent versions of scanpy save knn matrices in .obsp.
        In future versions of triku we will select these matrices from .obsp exclusively."""
        )

        knn_array = (
            object_triku.uns[  # type:ignore
                "neighbors"
            ][
                "connectivities"
            ]
            > 0
        ) + spr.identity(len(object_triku)).astype(
            bool
        )  # Saves memory

    return knn_array


def get_n_divisions(arr_counts: spr.csr.csr_matrix) -> int:
    diff = np.abs(
        (arr_counts - arr_counts.floor()).sum()
    )  # Faster .floor() than X.astype(int) (up to x2 for large arrays)
    triku_logger.log(
        TRIKU_LEVEL, f"Difference between int and float array is {diff}"
    )

    if diff < 1:
        n_divisions = 1
    else:
        n_divisions = 13  # Arbitrarily chosen. Prime numbers are better because 2, 3, 4 yield strange results during binning.

    triku_logger.log(TRIKU_LEVEL, f"Number of divisions set to {n_divisions}")
    return n_divisions


def return_knn_expression(
    arr_expression: spr.csr.csr_matrix, knn_indices: spr.csr.csr_matrix
) -> spr.csr.csr_matrix:
    """
    This function returns an array with the knn expression per gene and cell. To calculate the expression per gene
    we are going to apply the dot product of the neighbor indices and the expression.

    That is, if we have n_g as number of genes, and n_c as number of cells, the matrix product would be:

        Mask          Expr         Result
    (n_c x n_c) · (n_c x n_g) = (n_c x n_g)

    Then, the Result matrix would have in each cell, the summed expression of that gene in the knn (and also the
    own cell).

    In this step we do not mask the array. Previously, after the calculation of the expression, we masked the knn expression of the
    cells that were originally expressing that gene. That is, for any gene, the knn expression of the cells are were originally
    not expressing that gene was set to zero. We do that because we saw that not doing that produced "dirtier" EMD calculations.
    The thing is that since the matrices are csr, constructing a masked array requires a new matrix and selecting the elements from
    the knn matrix, or deleting the existing ones based on the count array, in both cases time consuming.

    To save that time, we will simply in the convolution step select the expression values with the mask for each gene, because that
    selection has to be done anyways.
    """

    knn_expression = knn_indices.dot(arr_expression)

    triku_logger.log(
        TRIKU_LEVEL,
        "knn_expression: {knn_expression} | {knn_expression.shape}",
    )

    return knn_expression


def compute_conv_idx(
    counts_gene: np.ndarray, knn: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a GENE x CELL matrix, and an index to select from, calculates the convolution of reads for that gene index.
    The function returns the
    """
    y_probs = np.bincount(counts_gene) / len(
        counts_gene
    )  # Important to transform count to probabilities
    # to keep the convolution constant.
    # THIS y_probs MUST have the probs of P(X=0) because the
    # random neighbors might have 0 counts!!!  [TODO: check this!!!]

    # We are calculating the convolution of cells with positive expression. Thus, in the first distribution
    # we have to remove the cells with 0 reads, and rescale the probabilities.
    arr_0 = y_probs.copy()
    arr_0[0] = 0
    arr_0 /= arr_0.sum()

    if (
        counts_gene.sum() > 7000
    ):  # This is important. For some genes, if the counts are too big, np.convolve crashes!!
        func = fftconvolve
    else:
        func = np.convolve

    arr_convolve = func(arr_0, y_probs)  # First iteration always with itself

    for _ in range(knn):
        arr_convolve = func(arr_convolve, y_probs,)

    arr_prob = (
        arr_convolve / arr_convolve.sum()
    )  # This is just in case the sum is bigger than 1

    x_conv, y_conv = np.arange(len(arr_prob)), arr_prob

    return x_conv, y_conv


def calculate_emd(
    knn_counts: np.ndarray,
    x_conv: np.ndarray,
    y_conv: np.ndarray,
    n_divisions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns "normalized" earth movers distance (EMD). The function calculates the x positions and probabilities
    of the "real" dataset using the knn_counts, and the x positions and probabilities of the convolution as attributes.

    To normalize the distance, it is divided by the standard deviation of the convolution. Since the convolution
    is already given as a distribution, mean and variance have to be calculated "by hand".
    """
    dist_range = np.arange(np.max(knn_counts) + 1)
    # np.bincount transforms [3, 3, 4, 1, 2, 9] into [0, 1, 1, 2, 1, 0, 0, 0, 0, 1]
    real_vals = np.bincount(knn_counts) / len(knn_counts)

    # IMPORTANT: either for std or emd calculation, all x variables must be scaled back!
    dist_range = dist_range / n_divisions
    x_conv = x_conv / n_divisions

    emd = sts.wasserstein_distance(dist_range, x_conv, real_vals, y_conv)

    mean = (x_conv * y_conv).sum()
    std = np.sqrt((y_conv * (x_conv - mean) ** 2).sum())

    return x_conv, emd / std


def compute_convolution_and_emd(
    array_counts: spr.csc.csc_matrix,
    array_knn_counts: spr.csc.csc_matrix,
    idx: int,
    knn: int,
    min_knn: int,
    n_divisions: int,
) -> np.ndarray:
    """Calculate the convolution and emd given the array with counts and with knn counts.
    To do the convolution we will select the gene column from each array.

    From the array of counts we will simply select the values, and from the array of knn counts we will select the values
    of the indices from the array of counts (arr_counts[:, idx].indices).

    Then, we are going to make the array integer. To do that, we recall the n_divisions argument, that applies
    binning to the unit. For instance, if the expression of a gene is 5.23 and 5 bins are set, the new expression is
    int(5.23 * 5) = int(26.15) = 26 -> 26 / 5 = 5.2 (so we lose 0.03 of expression).
    This is a scaling step

    """

    # counts_gene.indices returns the nonzero entries. To only call counts_gene once, we call it first as an sparse array,
    # then get the dense knn_counts, and finally, get the dense counts gene. This saves 30% time instead of setting
    # indices = array_counts[:, idx].indices first!!
    counts_gene = array_counts[:, idx]
    knn_counts = array_knn_counts[:, idx].T.A[0][counts_gene.indices]
    counts_gene = counts_gene.T.A[0]

    """
    1) We set indices because if we do array_counts[:, idx].T.A[0] we also get the zero elements!!!
    2) array_counts[:, idx].T.A[0][indices] is 3x faster than array_counts[indices, idx].T.A[0] because it is faster to set the row indexes
       in a dense array than in a csc array (remember that accesing columns in csr or rows in csc is extremely slow!!)
    3) .T is because the array is in column, and to get the dense version we transform to row: array([[0], [0], [0]]) -> array([[0, 0, 0]])
    4) .A makes the csc matrix to a dense array. since the array is 2D, the array has shape (1, X), so we do A[0] to set it to a 1d array

    Remember that ONLY THE KNN EXPRESSION FROM POSITIVELY-EXPRESSED CELLS is done!!!!
    """

    counts_gene = (counts_gene * n_divisions).astype(int)
    knn_counts = (knn_counts * n_divisions).astype(int)

    if len(counts_gene) < min_knn:
        emd = 0
    else:
        x_conv, y_conv = compute_conv_idx(counts_gene, knn)
        x_conv, emd = calculate_emd(knn_counts, x_conv, y_conv, n_divisions)

    return emd


def parallel_emd_calculation(
    array_counts: spr.csr.csr_matrix,
    array_knn_counts: spr.csr.csr_matrix,
    n_procs: int,
    knn: int,
    min_knn: int,
    n_divisions: int,
) -> Tuple[list, list, np.ndarray]:
    """
    Calculation of convolution for each gene, and its emd. To do that we call compute_convolution_and_emd which,
    in turn, calls compute_conv_idx to calculate the convolution of the reads; and calculate_emd, to calculate the
    emd between the convolution and the knn_counts.

    Since we are working with counts of each gene, instead of each cell, we will get the csc forms of array_knn_counts and array_counts.
    This conversion takes some time and memory, but it does save a lot of time afterwards, when doing the column indexing.
    e.g. with a 50000 x 10000 matrix, doing csr -> csc and csc indexing takes 8s, whereas doing csr indexing takes 30 mins!!

    To make things faster we use ray parallelization. Ray selects the counts and knn counts on each gene, and computes
    the convolution and distance. The output result is, for each gene, the convolution distribution
    (x, and probabilities), and the distances.
    """
    n_genes = array_counts.shape[1]

    triku_logger.log(
        TRIKU_LEVEL, f"Running EMD calulation with {n_procs} processors."
    )

    array_counts_csc = array_counts.tocsc()
    array_knn_counts_csc = array_knn_counts.tocsc()

    # Apply a non_paralellized variant with tqdm   ### TODO MAYBE REMOVE THE PARALLELIZED VERSION!!!
    if n_procs == 1:
        list_emd = [
            compute_convolution_and_emd(
                array_counts_csc,
                array_knn_counts_csc,
                idx_gene,
                knn,
                min_knn,
                n_divisions,
            )
            for idx_gene in range(n_genes)
        ]

    else:
        try:
            import ray
        except ImportError:
            raise ImportError(
                "Ray is not installed in the system. You can install it writing \npip install ray\n "
                "in the console. If you are on windows, ray might not be supported. Use n_comps=1 "
                "instead."
            )

        ray.shutdown()
        ray.init(num_cpus=n_procs, ignore_reinit_error=True)

        compute_convolution_and_emd_remote = ray.remote(
            compute_convolution_and_emd
        )
        array_counts_id = ray.put(array_counts_csc)
        array_knn_counts_id = ray.put(array_knn_counts_csc)

        ray_obj_ids = [
            compute_convolution_and_emd_remote.remote(
                array_counts_id,
                array_knn_counts_id,
                idx_gene,
                knn,
                min_knn,
                n_divisions,
            )
            for idx_gene in range(n_genes)
        ]

        triku_logger.log(TRIKU_LEVEL, "Parallel computation of distances.")
        list_emd = ray.get(ray_obj_ids)
        triku_logger.log(TRIKU_LEVEL, "Done.")

        del array_counts_id
        del array_knn_counts_id
        gc.collect()
        ray.shutdown()

    return np.array(list_emd)


def subtract_median(x, y, n_windows):
    """When working with EMD, we want to find genes with more deviation on emd compared with other genes with similar
    mean expression. With higher expressions EMD tends to increase. To reduce that basal level we will subtract the
    median EMD to the genes using a number of windows. The approach is quite reliable between 15 and 80 windows.

    Too many windows can over-normalize, and lose genes that have high emd but are alone in that window."""

    # We have to take the distance in logarithm to account for the wide expression ranges
    linspace = 10 ** np.linspace(
        np.min(np.log10(x)), np.max(np.log10(x)), n_windows + 1
    )
    y_adjust = y.copy()

    y_median_array = np.zeros(len(y))
    for i in range(n_windows):
        mask = (x >= linspace[i]) & (x <= linspace[i + 1])
        y_median_array[mask] = np.median(y[mask])

    y_adjust -= y_median_array

    return y_adjust


def get_cutoff_curve(y, s) -> float:
    """
    Plots a curve, and finds the best point by joining the extremes of the curve with a line, and selecting
    the point from the curve with the greatest distance.
    The distance between a point in a curve, and the straight line is set by the following equation
    if u,v is the point in the curve, and y = mx + b is the line, then
    x_opt = (u - mb + mv) / (1 + m^2)

    Here y attribute refers to the emd distances (after median subtraction preferably). Those distances are sorted,
    and ordered, and the curve is extracted from there.
    """

    min_y, max_y = np.min(y), np.max(y)
    m, b = (max_y - min_y) / len(y), min_y

    list_d = []

    for u, v in enumerate(np.sort(y)):
        x_opt = (u - m * b + m * v) / (1 + m ** 2)
        y_opt = x_opt * m + b
        d = (x_opt - u) ** 2 + (y_opt - v) ** 2

        list_d.append(d)

    # S is a corrector factor. It leverages the best value in the curve, and selects a more or less stringent
    # value in the curve. the maximum distance is multiplied by (1 - S), and the leftmost or rightmost index
    # is selected

    dist_s = (1 - np.abs(s)) * np.max(list_d)
    s_idx = np.argwhere(list_d >= dist_s)

    if s >= 0:
        max_d_idx = np.max(s_idx)
    else:
        max_d_idx = np.min(s_idx)

    return np.sort(y)[max_d_idx]
