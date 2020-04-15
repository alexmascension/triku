import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as spr

from ..genutils import get_cpu_count
from ._triku_functions import return_knn_indices, return_knn_expression
# old calls#
#todo: revise and remove
from ..pp import remove_outliers
from ..utils._triku_tl_utils import check_count_mat, check_null_genes
from ..tl._triku_functions import return_triku_gene_idx
from ..utils._triku_tl_entropy_utils import return_leiden_partitition, entropy_per_gene
from ..utils._general_utils import get_arr_counts_genes, save_triku, set_level_logger
from ..logg import logger

import warnings

warnings.filterwarnings('ignore')  # To ignore Numba warnings


def triku(object_triku: [sc.AnnData, pd.DataFrame], n_features=None, return_features=None, use_adata_knn=None,
          knn=None, s=-0.01, apply_background_correction=True, n_comps=50, metric='cosine', random_state=0,
          n_procs=None, verbose='info'):
    """
    This function calls the triku method using python directly. This function expects an
    annData object or a csv / txt matrix of n_cells x n_genes. The function then returns an array / list
    of the selected genes.

    Parameters
    ----------
    object_triku : scanpy.AnnData or pandas.DataFrame
        Object with count matrix. If `pandas.DataFrame`, rows are cells and columns are genes.
    n_features : int, None
        Number of features to select. If None, the number is chosen automatically.
    return_features : bool, None
        If True, returns the selected features as a list. It is always true if object_triku is of type pd.DataFrame
    use_adata_knn :  bool, None
        If object_triku is a scanpy.AnnData object, and sc.pp.neighbors was run, select neighbors and knn from
        adata.uns['neighbors']['connectivities'] and  adata.uns['neighbors']['params']['n_neighbors'].
    knn: int, None
        If use_adata_knn is False, number of neighbors to choose for feature selection. By default
        the half the square root of the number of cells is chosen.
    s : float
        Correction factor for automatic feature selection. Negative values imply a selction of more genes, and
        positive values imply a selection of fewer genes. We recommend values between -0.1 and 0.1.
    apply_background_correction : bool
        Substract the Wasserstein distance from a randomised adata to compensate the inflation of Wasserstein distance
        of highly expressed genes. If the dataset is too big, this step can be ommited, since those features usually
        don't get selected.
    n_comps : int
        Number of PCA components for knn selection.
    metric : str
        Metric for knn selection.
    random_state : int
        Seed for random processes
    n_procs : int, None
        Number of processes for parallel processing.
    verbose : str ['debug', 'info', 'warning', 'error', 'critical']
        Logger verbosity output.
    Returns
    -------
    list_features : list
        list of selected features
    """

    # Basic checks of variables
    set_level_logger(verbose)

    if isinstance(object_triku, pd.DataFrame):
        use_adata_knn = False

    if n_procs is None:
        n_procs = max(1, get_cpu_count() - 1)
    elif n_procs > get_cpu_count():
        logger.warning('The selected number of cpus ({}) is higher than the available number ({}). The number'
                       'of used cores will be set to {}.'.format(n_procs, get_cpu_count(), max(1, get_cpu_count() - 1)))
        n_procs = max(1, get_cpu_count() - 1)


    # Get the array of counts (np.array) and the array of genes. Additionally, return the a
    arr_counts, arr_genes = get_arr_counts_genes(object_triku)
    check_null_genes(arr_counts, arr_genes)
    check_count_mat(arr_counts)

    """
    First step is to get the kNN for the expression matrix.
    This is not that time intensive, but for reproducibility, we by default accept the kNN calculated by
    scanpy (sc.pp.neighbors()), and obtain the info from there. 
    Otherwise, we calculate the kNNs. 
    The expected output from this step is a matrix of cells x (kNN + 1), where each column includes the neighbor index
    of the cell number 
    """

    knn_array = None

    if isinstance(object_triku, sc.AnnData):
        if (use_adata_knn is None) | (use_adata_knn == True):
            if 'neighbors' in object_triku.uns:
                knn = object_triku.uns['neighbors']['params']['n_neighbors']
                logger.info('We found "neighbors" in the anndata, with knn={}'.format(knn))

                # Connectivities array contains a pairwise relationship between cells. We want to select, for
                # each cell, the knn "nearest" cells. We can easily do that with argsort. In the end we obtain a
                # cells x knn array with the top knn cells.
                knn_array = np.asarray(object_triku.uns['neighbors']['connectivities'].todense()
                                       ).argsort()[:, -knn::][::, ::-1]

                # Last step is to add a arange of 0 to n_cells in the first column.
                knn_array = np.concatenate((np.arange(knn_array.shape[0]).reshape(knn_array.shape[0], 1),
                                            knn_array), axis=1)

    if knn_array is None:
        if knn is None:
            knn = int(0.5 * (arr_counts.shape[0]) ** 0.5)
            logger.info('The number of neighbours by default will be {}'.format(knn))

        knn_array = return_knn_indices(arr_counts, knn=knn, return_random=False, random_state=random_state,
                                       metric=metric)


    # Calculate the expression in the kNN (+ own cell) for all genes
    arr_knn_expression = return_knn_expression(arr_counts, knn_array)


    # The same steps must be applied to a randomized expression count matrix if we must
    arr_counts_random, knn_array_random, arr_knn_expression_random = None, None, None

    if apply_background_correction:
        arr_counts_random = randomize_counts(arr_counts)
        knn_array_random = return_knn_indices(arr_counts, knn=knn, return_random=False, random_state=random_state,
                                       metric=metric)
        arr_knn_expression_random = return_knn_expression(arr_counts_random, knn_array_random)


    """
    Next step is to 
    """


