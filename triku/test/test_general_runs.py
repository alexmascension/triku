import time

import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

import triku as tk


selected_markers = [
    "CD79A",
    "CD14",
    "C3",
    "GZMB",
    "HMOX1",
    "ICAM4",
    "ITGA2B",
    "CLU",
]


@pytest.mark.general
def test_run_defaults():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    t = time.time()
    tk.tl.triku(adata)
    print("TT", time.time() - t)
    print(adata.var.loc[selected_markers])
    for pos_gene in selected_markers:
        print(pos_gene)
        assert adata.var["highly_variable"].loc[pos_gene]


@pytest.mark.general
def test_run_defaults_random():
    # Generate random neigbors to set Wasserstein distance to 0
    adata = sc.datasets.blobs(
        n_variables=5000, n_centers=5, cluster_std=2.0, n_observations=1000
    )
    adata.X = np.abs(adata.X)
    sc.pp.filter_cells(adata, min_genes=30)
    sc.pp.filter_genes(adata, min_cells=30)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15)

    adata_random = adata.copy()

    knn_random_mat = np.zeros((len(adata), len(adata)))
    for row in range(len(adata)):
        mat_options = list(np.arange(row)) + list(
            np.arange(row + 1, (len(adata)))
        )
        mat_idx = np.random.choice(mat_options, 15)
        knn_random_mat[row, mat_idx] = 1
    knn_random_mat = csr_matrix(knn_random_mat)

    adata_random.obsp["distances"], adata_random.obsp["connectivities"] = (
        knn_random_mat,
        knn_random_mat,
    )
    tk.tl.triku(adata)
    tk.tl.triku(adata_random)

    assert adata_random.var["triku_distance_uncorrected"].mean() < 0.1
    assert adata.var["triku_distance_uncorrected"].mean() > 1.5


@pytest.mark.end
def test_end():
    pass
