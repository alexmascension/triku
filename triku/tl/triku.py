# import logging
import warnings
from typing import Union

import numpy as np
import scanpy as sc

from ..logg import TRIKU_LEVEL
from ..logg import triku_logger
from ..utils._general_utils import set_level_logger
from ..utils._triku_tl_utils import get_arr_counts
from ._triku_functions import emd_calculation
from ._triku_functions import get_cutoff_curve
from ._triku_functions import get_n_divisions
from ._triku_functions import return_knn_array
from ._triku_functions import return_knn_expression
from ._triku_functions import subtract_median

warnings.filterwarnings("ignore")  # To ignore Numba warnings


def triku(
    object_triku: Union[sc.AnnData, str],
    n_features: Union[None, int] = None,
    use_raw: bool = True,
    n_divisions: Union[None, int] = None,
    s: Union[int, float] = -0.01,
    n_windows: int = 75,
    min_knn: int = 6,
    name: Union[str, None] = None,
    verbose: Union[None, str] = "warning",
) -> dict:  # type:ignore
    """
    This function calls the triku method using python directly. This function expects an
    annData object or a csv / txt matrix of n_cells x n_genes. The function then returns an array / list
    of the selected genes.

    Parameters
    ----------
    object_triku : scanpy.AnnData or pandas.DataFrame
        Object with count matrix. If `pandas.DataFrame`, rows are cells and columns are genes.
    n_features : int
        Number of features to select. If None, the number is chosen automatically.
    use_raw : bool
        If True, selects the adata.raw, if it exists.
        To set the .raw propety, set as: adata.raw = adata.
        This matrix is adjusted to select the genes and cells that
        appear in the current adata. E.g. if we are running triku with a subpopulation, triku will select the cells
        from adata.raw of that subpopulation. If certain genes have been removed after saving the raw, triku will not
        consider the removed genes.
    n_divisions : int, None
        If the array of counts is not integer, number of bins in which each unit will be divided to account for
        that effect. For example, if n_divisions is 10, then 0.12 and 0.13 would be in the same bin, and 0.12 and 0.34
        in two different bins. If n_divisions is 2, the two cases would be in the same bin.
        The higher the number of divisions the more precise the calculation of distances will be. It will be more
        time consuming, though. If n_divisions is None, we will adjust it automatically.
    s : float
        Correction factor for automatic feature selection. Negative values imply a selction of more genes, and
        positive values imply a selection of fewer genes. We recommend values between -0.1 and 0.1.
    n_windows : int
        Number of windows used for median subtraction of Wasserstein distance.
    min_knn : int
        minimum number of expressed cells based on the knn to apply the convolution. If a gene has less than min_knn
        expressing cells, Wasserstein distance is set to 0, and the convolution is set as the knn expression.
    name: str
        Name of the run. If None, stores results in "triku_X". Else, stores it in "triku_X_{name}".
    verbose : str ['debug', 'triku', 'info', 'warning', 'error', 'critical']
        Logger verbosity output.
    Returns
    -------
    list_features : list
        list of selected features
    """

    # Basic checks of variables and assertions!!!
    set_level_logger(verbose)

    for var in [
        n_features,
        n_windows,
        n_divisions,
    ]:
        assert (var is None) | (
            isinstance(var, int)
        ), f"The variable value {var} must be an integer!"

    if name is None:
        name_str = ""
    else:
        name_str = f"_{name}"

    # Check that neighbors are calculated. Else make the user calculate them!!!
    error_ms = "Neighbors not found in adata. Run sc.pp.neighbors() first."
    try:
        if "neighbors" not in object_triku.uns:  # type:ignore
            raise IndexError(error_ms)
    except AttributeError:
        raise IndexError(error_ms)

    # Assert that adata.X is sparse (warning to transform) and assert that gene names are unique.
    arr_counts = get_arr_counts(object_triku, use_raw=use_raw)

    # Get n_divisions if None:
    if n_divisions is None:
        n_divisions = get_n_divisions(arr_counts)

    """
    First step is to get the kNN for the expression matrix from the annData.
    The expected matrix from this step is a matrix of cells x kNN, where each column includes the neighbor index
    of the cell number. The matrix is sparse, so we will work with methods that adapt work with sparse matrices.
    """

    knn = object_triku.uns["neighbors"]["params"]["n_neighbors"]  # type:ignore

    # Boolean array showing the neighbors (including its own)
    knn_array = return_knn_array(object_triku)

    # Calculate the expression in the kNN (+ own cell) for all genes [CAUTION! This array is unmasked!!!! (more explained inside the funcion)]
    triku_logger.info("Calculating knn expression")
    arr_knn_expression = return_knn_expression(arr_counts, knn_array)

    # Set to csc matrix. This allows a faster selection by column, where the genes are located. Accession times improve up to 1000x!!!
    array_counts_csc = arr_counts.tocsc()
    array_knn_counts_csc = arr_knn_expression.tocsc()

    # Apply the convolution, and calculate the EMD. The convolution is quite fast, but we will still paralellize it.
    triku_logger.info("EMD calculation")
    triku_logger.log(TRIKU_LEVEL, "min_knn set to {}".format(min_knn))
    array_emd = emd_calculation(
        array_counts_csc=array_counts_csc,
        array_knn_counts_csc=array_knn_counts_csc,
        knn=knn,
        min_knn=min_knn,
        n_divisions=n_divisions,
    )

    triku_logger.info("Subtracting median")
    mean_counts = arr_counts.mean(0).A[0]

    array_emd_subt_median = subtract_median(
        x=mean_counts, y=array_emd, n_windows=n_windows
    )

    # Selection of best genes, either by the curve method or as the N highest ones.
    if n_features is None:
        triku_logger.info("Selecting cutoff point")
        dist_cutoff = get_cutoff_curve(y=array_emd_subt_median, s=s)
    else:
        dist_cutoff = np.sort(array_emd_subt_median)[-(n_features + 1)]
    triku_logger.info("Cutoff point set to {dist_cutoff}")

    # Returns phase. Return if object is not an adata or if return is set to true.
    # We set it like that so that scanpy knows which are the hvg.
    hvg = array_emd_subt_median > dist_cutoff
    object_triku.var[  # type:ignore
        "highly_variable"
    ] = hvg

    object_triku.var[  # type:ignore
        f"triku_distance{name_str}"
    ] = array_emd_subt_median

    object_triku.var[  # type:ignore
        f"triku_distance_uncorrected{name_str}"
    ] = array_emd

    object_triku.var[  # type:ignore
        f"triku_highly_variable{name_str}"
    ] = hvg

    if "triku_params" not in object_triku.uns:  # type:ignore
        object_triku.uns["triku_params"] = {}  # type:ignore

    object_triku.uns["triku_params"][name] = {  # type:ignore
        "knn": knn,
        "n_features": n_features,
        "s": s,
        "n_windows": n_windows,
        "min_knn": min_knn,
        "n_divisions": n_divisions,
    }
