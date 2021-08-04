import time

import pytest
import scanpy as sc

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
def test_run_defaults_one_core():
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


@pytest.mark.end
def test_end():
    pass
