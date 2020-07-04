import pytest

import scanpy as sc
import triku as tk
import numpy as np
import subprocess
import pandas as pd
import os


selected_markers = ["CD79A", "CD14", "C3", "GZMB", "HMOX1", "ICAM4", "ITGA2B", "CLU"]


@pytest.mark.general
def test_run_defaults():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    tk.tl.triku(adata)
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
        assert ret["highly_variable"][np.argwhere(adata.var_names == pos_gene)[0]]


@pytest.mark.general
def test_run_cli():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    df = adata.to_df()
    df.to_csv(os.getcwd() + "/sample_df_CLI.csv", sep=",")
    subprocess.run(["triku", f"os.getcwd()/sample_df_CLI.csv", "-verbose", "triku"])

    for ROOT, DIRS, FILES in os.walk(os.path.dirname(os.getcwd())):
        for file in FILES:
            if 'triku_return' in file:
                path = ROOT + '/' + file

    ret = pd.read_csv(path)
    for pos_gene in selected_markers:
        assert ret["highly_variable"][adata.var_names == pos_gene].values[0]


@pytest.mark.end
def test_end():
    pass
