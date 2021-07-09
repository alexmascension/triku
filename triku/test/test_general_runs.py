import os

import numpy as np
import pandas as pd
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
def test_run_defaults():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    tk.tl.triku(adata)
    for pos_gene in selected_markers:
        assert adata.var["highly_variable"].loc[pos_gene]


@pytest.mark.general
def test_run_defaults_random():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    tk.tl.triku(adata, apply_background_correction=True)
    for pos_gene in selected_markers:
        assert adata.var["highly_variable"].loc[pos_gene]


@pytest.mark.general
def test_run_defaults_multicore():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    tk.tl.triku(adata, n_procs=4, verbose="triku")
    print("Contents on adata.var: ", adata.var)
    for pos_gene in selected_markers:
        assert adata.var["highly_variable"].loc[pos_gene]


@pytest.mark.general
def test_run_dataframe():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    df = adata.to_df()
    ret = tk.tl.triku(df)
    for pos_gene in selected_markers:
        assert ret["highly_variable"][
            np.argwhere(adata.var_names == pos_gene)[0]
        ]


@pytest.mark.general
def test_run_cli():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    df = adata.to_df()
    df.to_csv(os.getcwd() + "/sample_df_CLI.csv", sep=",")
    tk.tl.triku(os.getcwd() + "/sample_df_CLI.csv", verbose="triku")

    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    print(f"We'll be looking for files at {root_dir}")
    for ROOT, DIRS, FILES in os.walk(root_dir):
        for file in FILES:
            if "triku_return" in file:
                path = ROOT + "/" + file

    ret = pd.read_csv(path)  # type: ignore
    for pos_gene in selected_markers:
        assert ret["highly_variable"][adata.var_names == pos_gene].values[0]


@pytest.mark.end
def test_end():
    pass
