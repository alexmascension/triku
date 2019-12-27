import scanpy as sc
import pandas as pd
import numpy as np

from ..pp import remove_outliers
from ..utils._triku_tl_utils import check_count_mat
from ..tl._triku_functions import return_triku_gene_idx, return_leiden_partitition
from ..logg import logger


def triku(object_triku: [sc.AnnData, pd.DataFrame], n_bins: int = 80, n_cycles: int = 4, s: float = 0,
          outliers: bool = False, sigma_remove_outliers: float = 5.0, delta_x: int = None, delta_y: int = None,
          random_state: int = 0, knn: int = None, resolution: float = 1.3, entropy_threshold: float = 0.98):
    """
    This function calls the triku method using python directly. This function expects an
    annData object or a csv / txt matrix of n_cells x n_genes. The function should then return an array / list
    of the selected genes.
    """

    if isinstance(object_triku, sc.AnnData):
        arr_counts = object_triku.X
        list_genes = object_triku.var_names
    elif isinstance(object_triku, pd.DataFrame):
        arr_counts = object_triku.values
        list_genes = object_triku.columns.values
    else:
        msg = "Accepted object types are scanpy annDatas or pandas DataFrames (columns are genes)."
        logger.error(msg)
        raise TypeError(msg)

    check_count_mat(arr_counts)

    if not outliers:
        logger.info('Removing outliers.')
        remove_outliers(arr_counts, sigma_remove_outliers)

    idx_selected_genes = return_triku_gene_idx(arr=arr_counts, n_bins=n_bins, n_cycles=n_cycles, s=s,
                                               delta_x=delta_x, delta_y=delta_y)

    '''
    The next step is to remove genes with high entropy. This high entropy can be considered as 1 or > 0.9X. In order
    to do that we first need to do a clustering on the dataset. Although the clustering biases the genes that will be
    removed, this part is quite robust, because the genes that we expect to remove have a low expression, and they do 
    not arrive to the minimum required expression. In that case, unless the clustering is really specific, those genes
    should always be removed.
    In order not to remove genes with local expression patterns, we will perform clustering with a high number of 
    clusters. We will choose a resolution for leiden of ~ 1.2, which produces a considerable number of clusters, and
    we have seen that works good for what we are looking for.
    '''

    leiden_partition = return_leiden_partitition(arr_counts, knn, random_state, resolution)
    n_clusters = len(leiden_partition)

    '''
    Once clusters are obtained, we calculate the proportion of non-zero expressing cells per clusters and per gene.
    With that, we obtain a histogram of proportions that we will use to establish a cut-off proportion, that is, 
    clusters with fewer expressing cells than the proportion (for each gene) will not be considered for entropy 
    calculation.
    '''

    list_proportions = []
    for gene_idx in range(arr_counts.shape[1]):
        for cluster in range(n_clusters):
            gene_cluster_cells = arr_counts[leiden_partition[cluster], gene_idx]
            list_proportions.append((gene_cluster_cells > 0).sum() / len(gene_cluster_cells))

    # Find the threshold
    hist, bin_edges = np.histogram(list_proportions, bins=135)
    bin_edges = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]

