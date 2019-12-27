import numpy as np

from ..tl._triku_functions import find_knee_point, savgol_filter

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


def entropy_proportion_threshold(arr: np.ndarray, dict_cluster_idx: dict, s_ent: float = 0.0):
    """
    This function returns the proportion of cells with expression per cluster should be removed.
    That is, given an unsupervised clustering, the proportion of cells expressing the gene is calculated, and
    added to a list. In the end, the distribution of proportions is calculated, and a threshold is returned. This
    threshold means that clusters with fewer cells expressing gene than the proportion will not be accounted to
    calculate the entropy of the gene.
    """

    list_proportions = []

    for gene in range(arr.shape[1]):
        for cluster in dict_cluster_idx.keys():
            cell_expression = arr[dict_cluster_idx[cluster], gene]
            prop_nonzero = (cell_expression > 0).sum() / len(cell_expression)
            list_proportions.append(prop_nonzero)

    y, bins = np.histogram(list_proportions, bins=125)
    x = [0.5*(bins[i] + bins[i+1]) for i in range(len(bins) - 1)]
    y_sg = savgol_filter(y, 2 * int(len(x) / 15) + 1, 3)

    idx_knee = find_knee_point(x, y_sg, s_ent)

    return x[idx_knee]


def entropy_per_gene(arr: np.array, list_genes: list, cluster_labels: np.ndarray, s_ent: float = 0.0):
    """
    Calculate the entropy per gene. For each gene, the proportion of cells expressing that gene in a cluster is calculated,
    and the clusters expressing less than a threshold are removed. For the rest of clusters, a normal expression per
    cluster is calculated, and this expression is used to calculate the entropy value. This entropy value is
    divided by the maximum entropy value. A dictionary with the entropy per gene is returned.
    """
    dict_cluster_idx = {i: np.argwhere(cluster_labels == i).flatten() for i in list(dict.fromkeys(cluster_labels))}
    n_clusters = len(dict.fromkeys(cluster_labels))
    max_entropy = np.log2(n_clusters)

    threshold = entropy_proportion_threshold(arr, dict_cluster_idx, s_ent)

    dict_entropy_genes, dict_proportions_genes = {}, {}

    for gene_idx in range(arr.shape[1]):
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
            list_proportions = list_proportions[list_proportions > 0] / np.sum(list_proportions)
            entropy = - np.sum(list_proportions * np.log2(list_proportions)) / max_entropy

        gene = list_genes[gene_idx]
        dict_entropy_genes[gene] = entropy
        dict_proportions_genes[gene] = list_proportions

    return dict_entropy_genes, dict_proportions_genes


