import numpy as np
import scipy.sparse as spr

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
    non_zero_counts = (mat != 0).sum(0)

    if isinstance(non_zero_counts, np.matrix):
        non_zero_counts = np.asarray(non_zero_counts)

    return non_zero_counts / n_cells


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

    mean_per_gene = np.mean(mat, axis=0)

    if isinstance(mean_per_gene, np.matrix):
        mean_per_gene = np.asarray(mean_per_gene)

    return mean_per_gene

