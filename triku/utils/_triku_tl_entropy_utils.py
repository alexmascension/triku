import numpy as np
import pandas as pd
import scipy.sparse as spr
from tqdm import tqdm
import scanpy as sc

from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors
from umap import UMAP
from sklearn.decomposition import PCA
import leidenalg
from leidenalg.VertexPartition import RBConfigurationVertexPartition
from scanpy.utils import get_igraph_from_adjacency

from ..tl._triku_functions import find_knee_point
from ..logg import logger


def return_PCA(arr_counts):
    logger.info("... reducing dimensions on PCA.")
    # To save time, we will do a PCA with 50 components, and get the kNN from there
    if spr.isspmatrix(arr_counts):
        pca = PCA(n_components=25, whiten=True, svd_solver='auto').fit_transform(arr_counts.todense())
    else:
        pca = PCA(n_components=25, whiten=True, svd_solver='auto').fit_transform(arr_counts)

    return pca


def return_leiden_partitition(arr_counts, knn, random_state, resolution, leiden_from_adata, adata):
    """
    Calculation of leiden partition. For practicality, we will load the leiden solution from the annData if
    possible. This will take much less time and, more interestingly, the user will be able to run directly with their
    leiden solution, which is less bothersome.
    """
    logger.info("Calculating leiden for entropy...")

    # Zero, return leiden solution from annData if it exists
    if leiden_from_adata and isinstance(adata, sc.AnnData):
        if 'leiden' in adata.obs:
            leiden_partition = adata.obs['leiden'].values

            logger.info("We have found a clustering solution in the AnnData object with resolution {} and {} clusters. "
                        "If you don't want to use this solution, call tl.triku(..., leiden_from_adata=False).".format(
                adata.uns['leiden']['params']['resolution'], len(set(leiden_partition))
            ))

            leiden_partition = leiden_partition.astype(type(leiden_partition[0]))

            return leiden_partition

        else:
            logger.info("""We have not found a leiden solution in the anndata object. You can get it running 
                        scanpy.tl.leiden(adata).""")

    # First, compute the kNN of the matrix. With those kNN we will generate the adjacency matrix and the graph
    if knn is None:
        knn = int(arr_counts.shape[0] ** 0.5)

    pca = return_PCA(arr_counts)

    logger.info("... obtaining the kNN.")
    knn_indices, knn_dists, forest = nearest_neighbors(pca, n_neighbors=knn, metric='cosine',
                               random_state=np.random.RandomState(random_state), angular=False, metric_kwds={})

    logger.info("... obtaining the adjacency matrix.")
    adj = fuzzy_simplicial_set(pca, knn, None, None, knn_indices=knn_indices, knn_dists=knn_dists, set_op_mix_ratio=1,
                               local_connectivity=1,)

    logger.info("... creating graph.")
    # Create Graph
    g = get_igraph_from_adjacency(adjacency=adj)
    weights = np.array(g.es['weight']).astype(np.float64)

    logger.info("... running leiden.")
    leiden_partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                                resolution_parameter=resolution, weights=weights, seed=random_state)
    leiden_partition = np.array(leiden_partition.membership)

    return leiden_partition


def entropy_proportion_threshold(arr_counts, leiden_partition, s):
    """
    This function returns the proportion of cells with expression per cluster should be removed.
    That is, given an unsupervised clustering, the proportion of cells expressing the gene is calculated, and
    added to a list. In the end, the distribution of proportions is calculated, and a threshold is returned. This
    threshold means that clusters with fewer cells expressing gene than the proportion will not be accounted to
    calculate the entropy of the gene.
    """
    list_proportions = []
    for cluster in list(leiden_partition.keys()):
        n_cells = len(leiden_partition[cluster])
        array_cluster = arr_counts[leiden_partition[cluster], :]
        proportion_cluster = (array_cluster > 0).sum(0) / n_cells
        list_proportions += proportion_cluster.tolist()

    # Find the threshold
    hist, bin_edges = np.histogram(list_proportions, bins=150)
    bin_edges = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]

    x = np.arange(len(bin_edges)) / len(bin_edges)
    y = (hist - min(hist)) / (np.max(hist) - np.min(hist))

    p = find_knee_point(x, y, s=s)

    return bin_edges[p]


def norm_entropy(arr: [np.ndarray]):
    """
    Calculates the normalized entropy of a given array. The array must contain all zeros, if necessary.
    The normalized entropy is defined as the entropy of the array divided by the maximum entropy, log2(len(array)).
    """
    if np.sum(arr) == 0:
        return 1
    else:
        arr_nonzero = arr[arr > 0]
        return - arr_nonzero * np.log2(arr_nonzero) / np.log2(len(arr))


def entropy_per_gene(arr: np.array, list_genes: list, cluster_labels: [np.ndarray, RBConfigurationVertexPartition],
                     s_ent: float = 0.0):
    """
    Calculate the entropy per gene. For each gene, the proportion of cells expressing that gene in a cluster is calculated,
    and the clusters expressing less than a threshold are removed. For the rest of clusters, a normal expression per
    cluster is calculated, and this expression is used to calculate the entropy value. This entropy value is
    divided by the maximum entropy value. A dictionary with the entropy per gene is returned.
    """
    logger.info("Calculating entropy per gene")
    if isinstance(cluster_labels, np.ndarray):
        dict_cluster_idx = {i: np.argwhere(cluster_labels == i).flatten() for i in list(dict.fromkeys(cluster_labels))}
    elif isinstance(cluster_labels, RBConfigurationVertexPartition):
        dict_cluster_idx = {i: np.array(cluster_labels[i]) for i in range(len(cluster_labels))}
    else:
        raise TypeError("cluster_labels is of type {} and must be of types numpy.ndarray or "
                        "RBConfigurationVertexPartition.".format(type(cluster_labels)))

    n_clusters = len(dict_cluster_idx)
    max_entropy = np.log2(n_clusters)

    threshold = entropy_proportion_threshold(arr, dict_cluster_idx, s_ent)
    logger.info('Threshold for minimum expresing cells per cluster set at {}.'.format(threshold))

    dict_entropy_genes, dict_proportions_genes, dict_percentage_counts_genes = {}, {}, {}

    for gene_idx in tqdm(range(arr.shape[1])):
        list_proportions, list_percentage_counts = [], []
        for cluster in dict_cluster_idx.keys():
            list_proportions.append(np.sum(arr[dict_cluster_idx[cluster], gene_idx]) /
                                    len(dict_cluster_idx[cluster]))
            list_percentage_counts.append((arr[dict_cluster_idx[cluster], gene_idx] > 0).sum() /
                                          len(dict_cluster_idx[cluster]))

        list_proportions, list_percentage_counts = np.array(list_proportions), np.array(list_percentage_counts)
        list_percentage_counts = list_percentage_counts[list_percentage_counts > threshold]

        if len(list_percentage_counts) == 0:
            entropy = 1
        else:
            list_proportions = list_proportions[list_proportions > 0]
            list_proportions = list_proportions / np.sum(list_proportions)
            entropy = - np.sum(list_proportions * np.log2(list_proportions)) / max_entropy

        gene = list_genes[gene_idx]
        dict_entropy_genes[gene] = entropy
        dict_proportions_genes[gene] = list_proportions
        dict_percentage_counts_genes[gene] = list_percentage_counts

    return dict_entropy_genes, dict_proportions_genes, dict_percentage_counts_genes
