import pytest

import scanpy as sc
import triku as tk
import numpy as np
from time import time


@pytest.fixture
def getpbmc3k():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=10)
    tk.tl.triku(adata)

    return adata


@pytest.fixture()
def run_adata_n_procs():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=10)

    dict_results, dict_times = {}, {}

    for n_procs in [1, 2, 4, 8]:
        t = time()
        tk.tl.triku(adata, n_procs=n_procs)
        dict_times[n_procs] = time() - t
        dict_results[n_procs] = adata.var['emd_distance'].values

    return dict_times, dict_results


@pytest.mark.calc_check
def test_n_features(getpbmc3k):
    adata = getpbmc3k
    for n_feats in [1, 50, 100, 500, 1000]:
        tk.tl.triku(adata, n_features=n_feats)
        assert np.sum(adata.var['highly_variable'].values) == n_feats


@pytest.mark.output_check
def test_output_n_procs(run_adata_n_procs):
    dict_times, dict_results = run_adata_n_procs
    for n_procs in [2, 4, 8]:
        assert np.all(dict_results[1] == dict_results[n_procs])


@pytest.mark.calc_check
def test_output_n_procs(run_adata_n_procs):
    dict_times, dict_results = run_adata_n_procs
    for n_procs in [2, 4, 8]:
        assert dict_times[n_procs] < 2 * dict_times[1]


@pytest.mark.output_check
def test_use_raw(getpbmc3k):
    adata = getpbmc3k
    emd_not_raw = adata.var['emd_distance']

    adata.raw = adata
    sc.pp.log1p(adata)
    tk.tl.triku(adata, use_raw=True)
    emd_raw = adata.var['emd_distance']

    assert np.all(emd_raw.values == emd_not_raw.values)


@pytest.mark.output_check
def test_do_return(getpbmc3k):
    adata = getpbmc3k
    ret = tk.tl.triku(adata, do_return=True, verbose="triku", n_procs=1)

    assert np.all(adata.var['highly_variable'].values == ret['highly_variable'].values)
    for name_col in ['emd_distance', 'emd_distance_uncorrected']:
        assert np.all(adata.var[name_col] == ret[name_col])


@pytest.mark.calc_check
def test_use_adata_knn(getpbmc3k):
    adata = getpbmc3k
    sc.pp.neighbors(adata, n_neighbors=25)
    tk.tl.triku(adata, n_procs=1)
