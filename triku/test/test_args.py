from time import time

import numpy as np
import pytest
import scanpy as sc

import triku as tk


@pytest.fixture
def getpbmc3k():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=10)
    tk.tl.triku(adata)
    adata.X = np.asarray(adata.X.todense())

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
        dict_results[n_procs] = adata.var["emd_distance"].values

    return dict_times, dict_results


@pytest.mark.calc_check
def test_n_features(getpbmc3k):
    adata = getpbmc3k
    for n_feats in [1, 50, 100, 500, 1000]:
        tk.tl.triku(adata, n_features=n_feats)
        assert np.sum(adata.var["highly_variable"].values) == n_feats


@pytest.mark.calc_check_parallel
def test_output_n_procs(run_adata_n_procs):
    dict_times, dict_results = run_adata_n_procs
    for n_procs in [2, 4, 8]:
        assert dict_times[n_procs] < 2 * dict_times[1]


@pytest.mark.calc_check
def test_use_adata_knn(getpbmc3k):
    adata = getpbmc3k
    sc.pp.neighbors(adata, n_neighbors=25)
    tk.tl.triku(adata, n_procs=1)


@pytest.mark.calc_check
def test_n_windows_1(getpbmc3k):
    adata = getpbmc3k
    tk.tl.triku(
        adata, n_procs=1, n_windows=1, apply_background_correction=True
    )
    assert not np.all(
        adata.var["emd_distance"].values
        == adata.var["emd_distance_uncorrected"].values
    )

    # Checks that previous columns are removed
    tk.tl.triku(
        adata, n_procs=1, n_windows=1, apply_background_correction=False
    )
    assert np.all(
        adata.var["emd_distance"].values
        == adata.var["emd_distance_uncorrected"].values
        - np.median(adata.var["emd_distance_uncorrected"].values)
    )

    tk.tl.triku(
        adata, n_procs=1, n_windows=100, apply_background_correction=False
    )
    assert (
        np.sum(
            adata.var["emd_distance"].values
            - adata.var["emd_distance_uncorrected"].values
        )
        < 0
    )


@pytest.mark.output_check
def test_use_raw(getpbmc3k):
    adata = getpbmc3k
    emd_not_raw = adata.var["emd_distance"]

    adata.raw = adata
    sc.pp.log1p(adata)
    tk.tl.triku(adata, use_raw=True)
    emd_raw = adata.var["emd_distance"]

    assert np.all(emd_raw.values == emd_not_raw.values)


@pytest.mark.output_check
def test_do_return(getpbmc3k):
    adata = getpbmc3k
    ret = tk.tl.triku(adata, do_return=True, verbose="triku", n_procs=1)

    assert np.all(adata.var["highly_variable"] == ret["highly_variable"])
    for name_col in ["emd_distance", "emd_distance_uncorrected"]:
        assert np.all(adata.var[name_col] == ret[name_col])


@pytest.mark.output_check
def test_bg_correction():
    adata = sc.datasets.pbmc3k_processed()
    tk.tl.triku(adata, apply_background_correction=False)
    assert "emd_distance_random" not in adata.var

    tk.tl.triku(adata, apply_background_correction=True)
    assert "emd_distance_random" in adata.var

    assert np.abs(np.mean(adata.var["emd_distance_random"].values)) < 0.1


@pytest.mark.output_check
def test_s():
    n_feats = []

    for s in [
        0.1,
        0.05,
        0,
        -0.05,
        -0.1,
    ]:  # This s order guarantees that the number of selected feats will be increasing
        adata = sc.datasets.pbmc3k_processed()
        tk.tl.triku(adata, s=s)
        n_feats.append(adata.var["highly_variable"].values.sum())

    assert sorted(n_feats) == n_feats


@pytest.mark.var_check
def test_n_divisions_check(getpbmc3k):
    adata = getpbmc3k
    assert np.sum(adata.X - adata.X.astype(int)) == 0

    adata.X = adata.X.astype(int)
    tk.tl.triku(adata, n_procs=1)
    assert adata.uns["triku_params"]["n_divisions"] == 1

    adata.X = adata.X.astype(float)
    tk.tl.triku(adata, n_procs=1)
    assert adata.uns["triku_params"]["n_divisions"] == 1

    sc.pp.log1p(adata)
    tk.tl.triku(adata, n_procs=1)
    assert adata.uns["triku_params"]["n_divisions"] > 1

    adata.X = np.ceil(adata.X).astype(int)
    tk.tl.triku(adata, n_procs=1)
    assert adata.uns["triku_params"]["n_divisions"] == 1
