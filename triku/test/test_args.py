import numpy as np
import pytest
import scanpy as sc

import triku as tk


@pytest.fixture
def getpbmc3k():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.neighbors(adata)

    # tk.tl.triku(adata)

    return adata


@pytest.mark.calc_check
def test_n_features(getpbmc3k):
    adata = getpbmc3k
    for n_feats in [1, 50, 100, 500, 1000]:
        tk.tl.triku(adata, n_features=n_feats)
        assert np.sum(adata.var["highly_variable"].values) == n_feats


@pytest.mark.output_check
def test_use_raw(getpbmc3k):
    adata = getpbmc3k
    tk.tl.triku(adata, use_raw=False)
    emd_not_raw = adata.var["triku_distance"]

    adata.raw = adata
    sc.pp.log1p(adata)
    tk.tl.triku(adata, use_raw=True)
    emd_raw = adata.var["triku_distance"]

    assert np.all(emd_raw.values == emd_not_raw.values)


@pytest.mark.output_check
def test_s(getpbmc3k):
    n_feats = []

    for s in [
        0.1,
        0.05,
        0,
        -0.05,
        -0.1,
    ]:  # This s order guarantees that the number of selected feats will be increasing
        adata = getpbmc3k
        tk.tl.triku(adata, s=s)
        n_feats.append(adata.var["highly_variable"].values.sum())

    assert sorted(n_feats) == n_feats


@pytest.mark.var_check
def test_n_divisions_check(getpbmc3k):
    adata = getpbmc3k
    assert np.sum(adata.X - adata.X.astype(int)) == 0

    adata.X = adata.X.astype(int)
    tk.tl.triku(adata)
    assert adata.uns["triku_params"][None]["n_divisions"] == 1

    adata.X = adata.X.astype(float)
    tk.tl.triku(adata)
    assert adata.uns["triku_params"][None]["n_divisions"] == 1

    sc.pp.log1p(adata)
    tk.tl.triku(adata)
    assert adata.uns["triku_params"][None]["n_divisions"] > 1


@pytest.mark.var_check
def test_names(getpbmc3k):
    adata = getpbmc3k

    tk.tl.triku(adata)
    tk.tl.triku(adata, name="sample")

    for var in [
        "triku_distance",
        "triku_distance_sample",
        "triku_distance_uncorrected",
        "triku_distance_uncorrected_sample",
    ]:
        assert var in adata.var

    assert None in adata.uns["triku_params"]
    assert "sample" in adata.uns["triku_params"]

    assert (
        adata.uns["triku_params"][None] == adata.uns["triku_params"]["sample"]
    )
    assert np.all(
        adata.var["triku_distance"] == adata.var["triku_distance_sample"]
    )
    assert np.all(
        adata.var["triku_distance_uncorrected"]
        == adata.var["triku_distance_uncorrected_sample"]
    )
