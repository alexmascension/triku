import pytest

import logging
import numpy as np
import time

import scanpy as sc
import triku as tk


@pytest.mark.general
def test_run_defaults():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    tk.tl.triku(adata)
    for pos_gene in ['CD79A', 'CD14', 'C3', 'GZMB', 'HMOX1', 'ICAM4', 'ITGA2B', 'CLU']:
        assert adata.var['highly_variable'].loc[pos_gene]


@pytest.mark.end
def test_end():
    pass

