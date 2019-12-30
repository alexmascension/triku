import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as spr

from ..pp import remove_outliers
from ..utils._triku_tl_utils import check_count_mat
from ..tl._triku_functions import return_triku_gene_idx
from ..utils._triku_tl_entropy_utils import return_leiden_partitition, entropy_per_gene
from ..utils._general_utils import get_arr_counts_genes, save_triku
from ..logg import logger

import warnings

warnings.filterwarnings('ignore')  # To ignore Numba warnings


def triku(object_triku: [sc.AnnData, pd.DataFrame, str], n_bins: int = 80, write_anndata: bool = True,
          n_cycles: int = 4, s: float = 0, outliers: bool = False, sigma_remove_outliers: float = 6.0,
          delta_x: int = None, delta_y: int = None, random_state: int = 0, knn: int = None,
          resolution: float = 1.3, entropy_threshold: float = 0.98, s_entropy: float = -0.01,
          save_dir='', save_name=''):
    """
    This function calls the triku method using python directly. This function expects an
    annData object or a csv / txt matrix of n_cells x n_genes. The function should then return an array / list
    of the selected genes.

    Parameters
    ----------
    object_triku : scanpy.AnnData or pandas.DataFrame
        Object with count matrix. If `pandas.DataFrame`, rows are cells and columns are genes.
        If str, path to the annData file or pandas DataFrame.
    n_bins : int
        Number of bins to divide the percentage of zeros. Each bin will contain a similar amount of genes.
    write_anndata : bool
        Writes the results of the analysis into object_triku. Creates the column `.var['triku_entropy']` which
        stores the entropy of each gene, and the column `.var['triku_selected_gene']` that indicates if the gene
        is selected or not.
    n_cycles : int
        For each bin `[a, b]` selects also bins `[a + i(b-a)/N, b + i(b-a)/N]` to select genes. This makes the selection
        of the genes along the mean VS 0 percentage more gradual. Best effects are obtained with values between 3 and 5.
    s : float
        Correction factor for gene selection. Fewer genes are selected with positive values of `s` and
        more genes are selected with negative values. We recommend values between -0.1 and 0.1.
    outliers : bool
        If `False` removes values of counts that are extreme. Values with more standard deviations that
        a certain value are changed to the mean expression of the gene.
    sigma_remove_outliers : float
        Number of standard deviations to assign a value as an outlier.
    delta_x : int
        Intermediate parameter for gene selection. When selecting the cut point for gene selection for a bin, some
        curves [rank VS mean] show a plateau that makes the threshold be less accurate. To correct that, the curve is
        cut at the plateau. delta_x and delta_y are the values of the sliding box to decide the point of the plateau.
        Smaller values of delta_x and delta_y imply more stringent selection of the plateau. We recommend not to alter
        this values.
    delta_y : int
        See delta_x
    random_state : int
        Seed for clustering used in entropy calculation.
    knn : int
        Number of neighbors used in clustering for entropy calculation. By default, it is `sqrt(n_cells)`.
    resolution : float
        Leiden resolution used in clustering for entropy calculation.
    entropy_threshold : float
        Discard genes with entropy higher than that threshold.
    s_entropy : float
        Correction factor, similar to `s`. This factor is applied to calculate a threshold to discard low-expressed
        genes. For each cluster if the proportion of cells expressing that
        gene is smaller than that threshold, the cluster is not considered. If none of the clusters passes that
        threshold the entropy of that gene is set to 1. Positive values of `s_entropy` imply more stringent thresholds,
        and fewer genes are selected. Recommended values are between -0.05 and 0.05.

    Returns
    -------
    dict_triku : dict
        `triku_selected_genes`: list with selected genes.
        `triku_entropy`: entropy for each gene (selected or not).
    """

    arr_counts, arr_genes = get_arr_counts_genes(object_triku)

    check_count_mat(arr_counts)

    if not outliers:
        logger.info('Removing outliers.')
        arr_counts = remove_outliers(arr_counts, sigma_remove_outliers, do_copy=True)

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

    '''
    Once clusters are obtained, we calculate the proportion of non-zero expressing cells per clusters and per gene.
    With that, we obtain a histogram of proportions that we will use to establish a cut-off proportion, that is, 
    clusters with fewer expressing cells than the proportion (for each gene) will not be considered for entropy 
    calculation. This part is calculated inside entropy_per_gene.
    Finally, we will get the dictionary dict_entropy_genes, with the entropy of each gene, dict_proportions_genes,
    with the proportions of counts per gene, and dict_percentage_counts_genes, with the proportion of cells with
    positive expression per gene.
    '''

    dict_entropy_genes, dict_proportions_genes, dict_percentage_counts_genes = \
        entropy_per_gene(arr=arr_counts,
                         list_genes=arr_genes,
                         cluster_labels=leiden_partition,
                         s_ent=s_entropy)

    positive_genes = arr_genes[idx_selected_genes]
    genes_good_entropy = [gene for gene in positive_genes if dict_entropy_genes[gene] <= entropy_threshold]

    dict_triku = {'triku_selected_genes': genes_good_entropy, 'triku_entropy': dict_entropy_genes}

    if isinstance(object_triku, sc.AnnData) and write_anndata:
        object_triku.var['triku_entropy'] = dict_entropy_genes.values()
        object_triku.var['triku_selected_genes'] = [True if i in genes_good_entropy else False for i in
                                                    object_triku.var_names]

    save_triku(dict_triku, save_dir, save_name, object_triku)

    return dict_triku
