import numpy as np
import scipy.sparse as spr
import bottleneck as bn

from triku.logg import logger

def return_proportion_zeros(mat: [np.ndarray, spr.csr.csr_matrix]):
    """
    Returns a 1D array with the percentages. We have to do it using methods both for sparse arrays
    and dense arrays, which limits the options to do it.

    Parameters
    ----------
    mat : [np.ndarray, scipy.sparse.csr_matrix, other sparse matrices]
        Array of cells x genes.

    Returns
    -------
    prop_zeros: np.1darray
        Array with proportion of zeros per gene
    """

    n_cells = mat.shape[0]
    zero_counts = (mat == 0).sum(0)

    if isinstance(zero_counts, np.matrix):
        zero_counts = np.asarray(zero_counts)
        if len(zero_counts) == 1:
            zero_counts = zero_counts.flatten()

    return zero_counts / n_cells


def return_mean(mat: [np.ndarray, spr.csr.csr_matrix]):
    """
    Returns a 1D array with the mean of the array. We have to do it using methods both for sparse arrays
    and dense arrays, which limits the options to do it.

    Parameters
    ----------
    mat : [np.ndarray, scipy.sparse.csr_matrix, other sparse matrices]
        Array of cells x genes.

    Returns
    -------
    prop_zeros: np.1darray
        Array with mean expression per gene.
    """

    mean_per_gene = bn.nanmean(mat, axis=0)

    if isinstance(mean_per_gene, np.matrix):
        mean_per_gene = np.asarray(mean_per_gene)
        if len(mean_per_gene) == 1:
            mean_per_gene = mean_per_gene.flatten()

    return mean_per_gene


def return_std(mat: [np.ndarray, spr.csr.csr_matrix]):
    """
    Returns a 1D array with the mean of the array. We have to do it using methods both for sparse arrays
    and dense arrays, which limits the options to do it.

    Parameters
    ----------
    mat : [np.ndarray, scipy.sparse.csr_matrix, other sparse matrices]
        Array of cells x genes.

    Returns
    -------
    prop_zeros: np.1darray
        Array with mean expression per gene.
    """

    mean_per_gene = bn.nanstd(mat, axis=0)

    if isinstance(mean_per_gene, np.matrix):
        mean_per_gene = np.asarray(mean_per_gene)
        if len(mean_per_gene) == 1:
            mean_per_gene = mean_per_gene.flatten()

    return mean_per_gene




def check_count_mat(mat: [np.ndarray, spr.csr.csr_matrix]):
    """
    This function outputs a warning if we suspect the matrix is in logarithm value
    """
    logger.info("Checking integrity of matrix.")

    n_factors = 0

    if np.min(mat) < 0:
        logger.warning("The count matrix contains negative values. Triku is supposed to run with raw count matrices.")

    if np.percentile(mat[mat > 0], 99.9) < 17:
        logger.warning("The count matrix looks normalized or log-transformed (percentile 99.9: {}). "
                       "Triku is supposed to run with raw count matrices.".format(np.percentile(mat[mat > 0], 99.9)))

    if mat.shape[1] > 25000:
        logger.warning("The count matrix contains more than 25000 genes. We recommend filtering some genes, up to "
                       "15000 - 18000 genes. You can do that in scanpy with the function 'sc.pp.filter_genes()'.")

    return n_factors


def check_null_genes(arr_counts: np.ndarray, arr_genes: np.ndarray):
    """
    Removes columns (genes) that have zero sum. These genes interfere in the analysis and are not useful at all.
    """
    logger.info("Checking zero-count genes.")

    idx = np.argwhere(arr_counts.sum(0) != 0).flatten()

    if len(idx) < arr_counts.shape[1]:
        logger.warning('There are {} genes ({} %) with no counts. These will be removed from the analysis.'.format(
            arr_counts.shape[1] - len(idx), int(100 * (arr_counts.shape[1] - len(idx)) / arr_counts.shape[1])))

    return arr_counts[:, idx], arr_genes[idx]
