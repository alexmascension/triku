import numpy as np
import pytest
import scanpy as sc

import triku as tk


@pytest.mark.exception_check
def test_triku_check_count_mat_20000_vars():
    adata = sc.datasets.blobs(
        n_variables=22000, n_centers=3, cluster_std=1, n_observations=500
    )
    adata.X = np.abs(adata.X).astype(int)
    print(adata.X)

    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=1)

    assert adata.X.shape[1] > 20000

    tk.tl.triku(adata)


@pytest.mark.exception_check
def test_triku_check_count_negative():
    adata = sc.datasets.blobs(
        n_variables=2000, n_centers=3, cluster_std=1, n_observations=500
    )
    assert np.min(adata.X) < 0

    try:
        tk.tl.triku(adata)
    except BaseException:
        pass
    else:
        raise BaseException


@pytest.mark.exception_check
def test_triku_check_null_genes():
    adata = sc.datasets.blobs(
        n_variables=2000, n_centers=3, cluster_std=1, n_observations=500
    )
    adata.X = np.abs(adata.X).astype(int)
    adata.X[:, 1] = 0

    try:
        tk.tl.triku(adata)
    except BaseException:
        pass
    else:
        raise BaseException


@pytest.mark.exception_check
def test_triku_check_nonunique_varnames():
    adata = sc.datasets.blobs(
        n_variables=2000, n_centers=3, cluster_std=1, n_observations=500
    )
    adata.X = np.abs(adata.X).astype(int)
    adata.var_names = [0] + list(np.arange(len(adata.var_names) - 1))

    try:
        tk.tl.triku(adata)
    except BaseException:
        pass
    else:
        raise BaseException


@pytest.mark.exception_check
def test_triku_check_nonaccepted_type():
    adata = sc.datasets.blobs(
        n_variables=2000, n_centers=3, cluster_std=1, n_observations=500
    )
    adata.X = np.abs(adata.X).astype(int)
    adata.var_names = [0] + list(np.arange(len(adata.var_names) - 1))

    try:
        tk.tl.triku(adata.X)
    except BaseException:
        pass
    else:
        raise BaseException
