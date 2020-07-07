import bottleneck as bn
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as spr

from triku.logg import TRIKU_LEVEL
from triku.logg import triku_logger


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

    triku_logger.log(
        TRIKU_LEVEL,
        "zero_counts stats || min: {} | mean: {} |  max: {} | std: {}]".format(
            np.min(zero_counts),
            np.mean(zero_counts),
            np.max(zero_counts),
            np.std(zero_counts),
        ),
    )
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

    triku_logger.log(
        TRIKU_LEVEL,
        "mean stats || min: {} | mean: {} |  max: {} | std: {}]".format(
            np.min(mean_per_gene),
            np.mean(mean_per_gene),
            np.max(mean_per_gene),
            np.std(mean_per_gene),
        ),
    )
    return mean_per_gene


def check_count_mat(mat: [np.ndarray, spr.csr.csr_matrix]):
    """
    This function outputs a warning if we suspect the matrix is in logarithm value
    """
    triku_logger.info("Checking integrity of matrix.")

    if np.min(mat) < 0:
        error_msg = "The count matrix contains negative values. Triku is supposed to run with positive matrices."

        triku_logger.error(error_msg)
        raise BaseException(error_msg)

    if mat.shape[1] > 20000:
        triku_logger.warning(
            "The count matrix contains more than 25000 genes. We recommend filtering some genes, up to "
            "15000 - 18000 genes. You can do that in scanpy with the function 'sc.pp.filter_genes()'."
        )


def check_null_genes(arr_counts: np.ndarray):
    """
    Removes columns (genes) that have zero sum. These genes interfere in the analysis and are not useful at all.
    """
    triku_logger.info("Checking zero-count genes.")

    if np.any((arr_counts.sum(0) == 0).flatten()):
        error_msg = (
            "There are genes with no counts. Remove those genes first. "
            "You can use sc.pp.filter_genes(adata, min_cells=5)."
        )

        triku_logger.error(error_msg)
        raise BaseException(error_msg)


def check_adata_log1p(adata):
    if "log1p" in adata.uns:
        triku_logger.warning('We have found ªlop1p" in the ')


def make_genes_unique(arr):
    labels, counts = np.unique(arr, return_counts=True)
    non_unique_labels = labels[counts > 1]

    if len(non_unique_labels) > 0:
        msg_err = (
            "There are non-unique variable names. Make them unique by setting adata.var_names_make_unique() and"
            "run triku again."
        )
        triku_logger.error(msg_err)

        raise BaseException(msg_err)


def return_arr_counts_genes(object_triku, get_from_raw=None):
    triku_logger.log(TRIKU_LEVEL, "Obtaining count matrix and gene list.")
    # Check type of object and return the matrix as corresponded

    if isinstance(object_triku, sc.AnnData):
        if get_from_raw:
            triku_logger.info(
                "Using raw matrix. If you want to use the current matrix, set use_raw=False (although "
                "we discourage it)."
            )
            arr_counts = object_triku.raw[:, object_triku.var_names].X
            arr_genes = object_triku.var_names.values
        else:
            arr_counts, arr_genes = (
                object_triku.X,
                object_triku.var_names.values,
            )
    elif isinstance(object_triku, pd.DataFrame):
        arr_counts, arr_genes = (
            object_triku.values,
            object_triku.columns.values,
        )
    else:
        msg = "Accepted object types are scanpy annDatas or pandas DataFrames (columns are genes)."
        triku_logger.error(msg)
        raise TypeError(msg)

    if spr.isspmatrix(arr_counts):
        arr_counts = arr_counts.todense()

    if isinstance(arr_counts, np.matrix):
        arr_counts = np.asarray(arr_counts)

    make_genes_unique(arr_genes)

    triku_logger.log(
        TRIKU_LEVEL,
        "Array of counts\n{}, shape:{}".format(arr_counts, arr_counts.shape),
    )
    triku_logger.log(TRIKU_LEVEL, "Array of genes\n{}".format(arr_genes))
    return arr_counts, arr_genes


def get_arr_counts_and_genes(object_triku, use_raw):
    # Process the use_raw argument, and return the matrix of cells and genes accordingly.
    if isinstance(object_triku, sc.AnnData):
        if use_raw:
            if object_triku.raw is not None:
                arr_counts, arr_genes = return_arr_counts_genes(
                    object_triku, get_from_raw=True
                )
            else:
                check_adata_log1p(object_triku)
                arr_counts, arr_genes = return_arr_counts_genes(
                    object_triku, get_from_raw=False
                )

        else:
            check_adata_log1p(object_triku)
            arr_counts, arr_genes = return_arr_counts_genes(
                object_triku, get_from_raw=False
            )

        check_count_mat(arr_counts)
        check_null_genes(arr_counts)
    else:
        arr_counts, arr_genes = return_arr_counts_genes(object_triku)
        check_count_mat(arr_counts)
        check_null_genes(arr_counts)

    return arr_counts, arr_genes
