import numpy as np
import scipy.sparse as spr

from triku.logg import TRIKU_LEVEL
from triku.logg import triku_logger


def check_count_mat(mat: spr.csr.csr_matrix):
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


def return_arr_counts(object_triku, get_from_raw=None):
    triku_logger.log(TRIKU_LEVEL, "Obtaining count matrix and gene list.")
    # Check type of object and return the matrix as corresponded

    if get_from_raw:
        triku_logger.info(
            "Using raw matrix. If you want to use the current matrix, set use_raw=False (although "
            "we discourage it)."
        )
        arr_counts = object_triku.raw.X
    else:
        arr_counts = object_triku.X

    if not spr.isspmatrix(arr_counts):
        triku_logger.warning(
            "X is dense. We will set the matrix to sparse format (csr_matrix)."
        )
        arr_counts = spr.csr_matrix(
            arr_counts
        )  # We will require the array of counts to be sparse

        if get_from_raw:
            triku_logger.warning(
                "Can't set adata.raw.X to sparse. You can do it before setting the raw as adata.X = scipy.sparse.csr.csr_matrix(adata.X)"
            )
        else:
            object_triku.X = arr_counts

    triku_logger.log(
        TRIKU_LEVEL, "Array of counts\n{arr_counts}, shape:{arr_counts.shape}",
    )
    triku_logger.log(TRIKU_LEVEL, "Array of genes\n{arr_genes}")
    return arr_counts


def get_arr_counts(object_triku, use_raw):
    # Process the use_raw argument, and return the matrix of cells and genes accordingly.
    if use_raw:
        if object_triku.raw is not None:
            arr_counts = return_arr_counts(object_triku, get_from_raw=True)
        else:
            arr_counts = return_arr_counts(object_triku, get_from_raw=False)

    else:
        arr_counts = return_arr_counts(object_triku, get_from_raw=False)

    check_count_mat(arr_counts)
    check_null_genes(arr_counts)

    return arr_counts
