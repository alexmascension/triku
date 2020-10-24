import gc
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import scanpy as sc
import scipy.sparse as spr
import seaborn as sns
from matplotlib.lines import Line2D
from scikit_posthocs import (
    posthoc_quade,
)  # posthoc_nemenyi, posthoc_nemenyi_friedman
from scipy.stats import f
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARS
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

import triku as tk

try:
    from palettes_and_cmaps import prism
except ModuleNotFoundError:
    from .palettes_and_cmaps import prism


def clustering_binary_search(
    adatax,
    min_res,
    max_res,
    max_depth,
    seed,
    n_target_c=0,
    features=[],
    apply_log=None,
    transform_adata=False,
    res=1,
):
    depth = 0

    if not transform_adata:
        adata = adatax.copy()
    else:
        adata = adatax

    if apply_log is not None:
        if apply_log:
            sc.pp.log1p(adata)
    else:
        if "log1p" not in adata.uns:
            sc.pp.log1p(adata)

    adata.var["highly_variable"] = [i in features for i in adata.var_names]

    sc.pp.pca(adata, n_comps=35, use_highly_variable=True)
    sc.pp.neighbors(
        adata,
        n_neighbors=int(0.5 * len(adata) ** 0.5),
        random_state=seed,
        metric="cosine",
    )
    if n_target_c != 0:
        while depth < max_depth:
            #         print(f'Depth: {depth}, min res: {min_res}, max res: {max_res}')
            if depth == 0:
                sc.tl.leiden(adata, resolution=min_res, random_state=seed)
                leiden_sol, res_sol = adata.obs["leiden"], min_res
                if len(list(dict.fromkeys(leiden_sol))) == n_target_c:
                    break

                sc.tl.leiden(adata, resolution=max_res, random_state=seed)
                leiden_sol, res_sol = adata.obs["leiden"], max_res
                if len(list(dict.fromkeys(leiden_sol))) == n_target_c:
                    break

            mid_res = 0.5 * (max_res + min_res)
            sc.tl.leiden(adata, resolution=mid_res, random_state=seed)
            leiden_sol, res_sol = adata.obs["leiden"], mid_res
            n_clust_mid = len(list(dict.fromkeys(leiden_sol)))
            if n_clust_mid == n_target_c:
                break

            if n_clust_mid > n_target_c:
                max_res = mid_res
            else:
                min_res = mid_res

            depth += 1
    else:
        sc.tl.leiden(adata, resolution=res, random_state=seed)
        leiden_sol, res_sol = adata.obs["leiden"], res

    if not transform_adata:
        del adata
        gc.collect()

    return leiden_sol, res_sol


def get_max_diff_gene(
    adata, gene, group_col, per_expressing_cells=0.25, trim=0.025
):
    mean_exp_val, mean_exp_val_temp = [], []

    groups = sorted(list(set(adata.obs[group_col].values)))
    dict_argwhere = {
        g: np.argwhere(adata.obs[group_col].values == g).ravel()
        for g in groups
    }
    if spr.issparse(adata.X):
        exp_gene = np.asarray(adata[:, gene].X.todense()).ravel()
    else:
        exp_gene = adata[:, gene].X.ravel()

    # We will first exclude genes that do not pass a minimum expression threshold in any cluster
    for g in groups:
        exp_group = np.sort(exp_gene[dict_argwhere[g]].ravel())
        mean_exp_val_temp.append(
            np.mean(
                exp_group[: int((1 - per_expressing_cells) * len(exp_group))]
            )
        )  # for genes with small expression it may amplify noise
        mean_exp_val.append(
            np.mean(
                exp_group[
                    int(trim * len(exp_group)) : int(
                        (1 - trim) * len(exp_group)
                    )
                ]
            )
        )

    if np.all(np.isnan(mean_exp_val_temp)) or np.sum(mean_exp_val_temp) == 0:
        return 0, [0] * len(groups)

    mean_exp_val_to_1 = np.array(mean_exp_val) / sum(mean_exp_val)

    info = max(
        np.sort(mean_exp_val_to_1)[3] - np.sort(mean_exp_val_to_1)[0],
        np.sort(mean_exp_val_to_1)[-1] - np.sort(mean_exp_val_to_1)[-4],
    )

    return info, mean_exp_val_to_1


def plot_max_var_x_method(
    df_feature_ranks,
    df_max_var_dataset,
    feature_list=[0, 50, 100, 200, 500, 1000],
    title="",
    file="",
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    palette = prism

    for method_idx in range(len(df_feature_ranks.columns)):
        df_feature_ranks_method = df_feature_ranks[
            df_feature_ranks.columns[method_idx]
        ]
        for feat_idx in range(len(feature_list) - 2, -1, -1):
            genes_method_feat = df_feature_ranks_method[
                (df_feature_ranks_method >= feature_list[feat_idx])
                & (df_feature_ranks_method < feature_list[feat_idx + 1])
            ].index
            y_vals = df_max_var_dataset.loc[
                genes_method_feat, "maximum_variation"
            ].values
            ax.scatter(
                [method_idx + (len(feature_list) // 2 - feat_idx) * 0.125]
                * len(y_vals),
                y_vals,
                color=palette[feat_idx],
                s=15,
                alpha=0.35,
            )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=palette[idx],
            label=feature_list[idx + 1],
        )
        for idx in range(len(feature_list) - 1)
    ]
    ax.legend(handles=legend_elements)
    ax.set_xticks(range(len(df_feature_ranks.columns)))
    ax.set_xticklabels(df_feature_ranks.columns, rotation=45)
    ax.set_ylabel("Maximum difference")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.suptitle(title)

    for fmt in ["png", "pdf"]:
        fig.savefig(
            f"{os.getcwd()}/figures/comparison_figs/{fmt}/{file}.{fmt}",
            bbox_inches="tight",
        )


def plot_max_var_x_dataset(
    dict_df_feature_ranks,
    dict_df_max_var_dataset,
    n_features=200,
    title="",
    file="",
):
    # keys in dict_df_feature_ranks and in dict_df_feature_ranks must be in the same order!
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    palette = prism

    list_datasets = list(dict_df_feature_ranks.keys())
    method_list = dict_df_feature_ranks[list_datasets[0]].columns

    for x_idx, dataset in enumerate(list_datasets):
        df_feature_ranks = dict_df_feature_ranks[dataset]
        df_max_var_dataset = dict_df_max_var_dataset[dataset]

        for met_idx, method in enumerate(method_list[::-1]):
            df_method_rank = df_feature_ranks[method]
            genes_method_feat = df_method_rank[
                (df_method_rank >= 0) & (df_method_rank < n_features)
            ].index
            y_vals = df_max_var_dataset.loc[
                genes_method_feat, "maximum_variation"
            ].values
            x = x_idx + (len(method_list) // 2 - met_idx) * 0.085

            ax.scatter(x, np.median(y_vals), color=palette[met_idx], s=13)
            ax.plot(
                [x, x],
                [np.percentile(y_vals, 25), np.percentile(y_vals, 75)],
                color=palette[met_idx],
                linewidth=2,
            )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=palette[idx],
            label=method_list[::-1][idx],
        )
        for idx in range(len(method_list))[::-1]
    ]
    ax.legend(handles=legend_elements)
    ax.set_xticks(range(len(list_datasets)))
    ax.set_xticklabels(list_datasets, rotation=45)
    ax.set_ylabel("Maximum difference")
    ax.set_xlabel("DE probability")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.suptitle(title)
    for fmt in ["png", "pdf"]:
        fig.savefig(
            f"{os.getcwd()}/figures/comparison_figs/{fmt}/{file}.{fmt}",
            bbox_inches="tight",
        )


def plot_ARI_x_method(dict_ARI, title="", figsize=(15, 8), file=""):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    palette = prism

    list_dfs = list(dict_ARI.keys())
    list_methods = dict_ARI[list_dfs[0]].columns

    for x_idx, method in enumerate(list_methods):
        for df_idx, df in enumerate(list_dfs):
            y_vals = dict_ARI[df][method].values
            median, per25, per75 = (
                np.median(y_vals),
                np.percentile(y_vals, 25),
                np.percentile(y_vals, 75),
            )
            w = 0.09

            x = x_idx + (len(list_dfs) // 2 - df_idx) * w

            ax.scatter([x] * len(y_vals), y_vals, color=palette[df_idx], s=13)
            ax.plot([x, x], [per25, per75], color=palette[df_idx], linewidth=2)
            ax.plot(
                [x - w / 2, x + w / 2],
                [median, median],
                color=palette[df_idx],
                linewidth=2,
            )

    legend_elements = [
        Line2D([0], [0], marker="o", color=palette[idx], label=list_dfs[idx])
        for idx in range(len(list_dfs))
    ]
    ax.legend(handles=legend_elements)
    ax.set_xticks(range(len(list_methods)))
    ax.set_xticklabels(list_methods, rotation=45)
    ax.set_ylabel("ARI")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.suptitle(title)
    for fmt in ["png", "pdf"]:
        fig.savefig(
            f"{os.getcwd()}/figures/comparison_figs/{fmt}/{file}.{fmt}",
            bbox_inches="tight",
        )


def plot_ARI_x_dataset(dict_ARI, title="", figsize=(15, 8), file=""):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    palette = prism

    list_dfs = list(dict_ARI.keys())
    list_methods = dict_ARI[list_dfs[0]].columns

    for df_idx, df in enumerate(list_dfs):
        for x_idx, method in enumerate(list_methods):
            y_vals = dict_ARI[df][method].values
            median, per25, per75 = (
                np.median(y_vals),
                np.percentile(y_vals, 25),
                np.percentile(y_vals, 75),
            )
            w = 0.09

            x = df_idx + (len(list_dfs) // 2 - x_idx) * w

            ax.scatter([x] * len(y_vals), y_vals, color=palette[x_idx], s=13)
            ax.plot([x, x], [per25, per75], color=palette[x_idx], linewidth=2)
            ax.plot(
                [x - w / 2, x + w / 2],
                [median, median],
                color=palette[x_idx],
                linewidth=2,
            )

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color=palette[idx], label=list_methods[idx]
        )
        for idx in range(len(list_methods) - 1, -1, -1)
    ]
    ax.legend(handles=legend_elements)
    ax.set_xticks(range(len(list_dfs)))
    ax.set_xticklabels(list_dfs, rotation=45)
    ax.set_ylabel("ARI")
    ax.set_xlabel("DE probability")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.suptitle(title)
    for fmt in ["png", "pdf"]:
        fig.savefig(
            f"{os.getcwd()}/figures/comparison_figs/{fmt}/{file}.{fmt}",
            bbox_inches="tight",
        )


def biological_silhouette_ARI_table(
    adata,
    df_rank,
    outdir,
    file_root,
    seed,
    cell_types_col="cell_types",
    n_procs=None,
    res=1,
    include_all_random=True,
):
    list_methods = ["triku"] + [i for i in df_rank.columns if "triku" not in i]

    if include_all_random:
        list_methods += ["all", "random"]

    df_score = pd.DataFrame(
        index=[
            "ARI",
            "ARI_random",
            "NMI",
            "NMI_random",
            "Sil_bench_UMAP",
            "Sil_bench_PCA",
            "Sil_bench_all_hvg",
            "Sil_bench_all_base",
            "Sil_bench_PCA_random",
            "Sil_bench_all_hvg_random",
            "Sil_bench_all_base_random",
            "Sil_leiden_UMAP",
            "Sil_leiden_PCA",
            "Sil_leiden_all_hvg",
            "Sil_leiden_all_base",
            "Sil_leiden_PCA_random",
            "Sil_leiden_all_hvg_random",
            "Sil_leiden_all_base_random",
            "DavBo_bench_PCA",
            "DavBo_bench_all_hvg",
            "DavBo_bench_all_base",
            "DavBo_bench_PCA_random",
            "DavBo_bench_all_hvg_random",
            "DavBo_bench_all_base_random",
            "DavBo_leiden_PCA",
            "DavBo_leiden_all_hvg",
            "DavBo_leiden_all_base",
            "DavBo_leiden_PCA_random",
            "DavBo_leiden_all_hvg_random",
            "DavBo_leiden_all_base_random",
        ],
        columns=list_methods,
    )
    cell_types = (
        adata.obs[cell_types_col].values
        if cell_types_col is not None
        else None
    )

    # Get number of HVG with triku.
    tk.tl.triku(adata, random_state=seed, n_procs=n_procs)
    n_hvg = np.sum(adata.var["highly_variable"])

    adata_copy = adata.copy()

    if "log1p" not in adata_copy.uns:
        sc.pp.log1p(adata_copy)

    for method in list_methods:
        if (method != "triku") & (method not in ["all", "random"]):
            adata_copy.var["highly_variable"] = [
                i in df_rank[method].sort_values().index.values[:n_hvg]
                for i in adata_copy.var_names
            ]
        elif method == "all":
            adata_copy.var["highly_variable"] = [True] * len(
                adata_copy.var_names
            )
        elif method == "random":
            array_selection = np.array([False] * len(adata_copy.var_names))
            array_selection[
                np.random.choice(
                    np.arange(len(adata_copy.var_names)), n_hvg, replace=False
                )
            ] = True
            adata_copy.var["highly_variable"] = array_selection
        else:  # method is triku_XXX
            adata_copy.var["highly_variable"] = [
                i in df_rank["triku_0"].sort_values().index.values[:n_hvg]
                for i in adata_copy.var_names
            ]

        features = adata_copy.var[
            adata_copy.var["highly_variable"]
            == True  # noqa    # Revert to == True if it fails !!!!!!
        ].index.values

        if cell_types is not None:
            leiden_sol, res = clustering_binary_search(
                adata_copy,
                0.1,
                2,
                7,
                seed=seed,
                n_target_c=len(set(cell_types)),
                features=features,
                apply_log=False,
                transform_adata=True,
            )
        else:
            leiden_sol, res = clustering_binary_search(
                adata_copy,
                0.1,
                2,
                7,
                seed=seed,
                n_target_c=0,
                features=features,
                apply_log=False,
                transform_adata=True,
                res=res,
            )

        sc.tl.umap(adata_copy)

        if cell_types is not None:
            leiden_sol_random = leiden_sol.copy()
            np.random.shuffle(leiden_sol_random)
            cell_types_random = cell_types.copy()
            np.random.shuffle(cell_types_random)

            ARI, NMIs = (
                ARS(leiden_sol, cell_types),
                NMI(leiden_sol, cell_types),
            )
            ARI_random, NMI_random = (
                ARS(leiden_sol_random, cell_types),
                NMI(leiden_sol_random, cell_types),
            )
            adata_copy.obs["leiden"] = leiden_sol
        else:
            ARI, NMIs, ARI_random, NMI_random = None, None, None, None

        if cell_types is not None:
            Sil_bench_UMAP = silhouette_score(
                adata_copy.obsm["X_umap"], cell_types, metric="cosine"
            )
            Sil_bench_PCA = silhouette_score(
                adata_copy.obsm["X_pca"], cell_types, metric="cosine"
            )
            Sil_bench_all_hvg = silhouette_score(
                adata_copy.X[:, adata_copy.var["highly_variable"].values],
                cell_types,
                metric="cosine",
            )
            Sil_bench_all_base = silhouette_score(
                adata_copy.X, cell_types, metric="cosine"
            )

            Sil_bench_PCA_random = silhouette_score(
                adata_copy.obsm["X_pca"], cell_types_random, metric="cosine"
            )
            Sil_bench_all_hvg_random = silhouette_score(
                adata_copy.X[:, adata_copy.var["highly_variable"].values],
                cell_types_random,
                metric="cosine",
            )
            Sil_bench_all_base_random = silhouette_score(
                adata_copy.X, cell_types_random, metric="cosine"
            )
        else:
            (
                Sil_bench_UMAP,
                Sil_bench_PCA,
                Sil_bench_all_hvg,
                Sil_bench_all_base,
            ) = (
                None,
                None,
                None,
                None,
            )
            (
                Sil_bench_PCA_random,
                Sil_bench_all_hvg_random,
                Sil_bench_all_base_random,
            ) = (None, None, None)

        Sil_leiden_UMAP = silhouette_score(
            adata_copy.obsm["X_umap"],
            adata_copy.obs["leiden"].values,
            metric="cosine",
        )
        Sil_leiden_PCA = silhouette_score(
            adata_copy.obsm["X_pca"],
            adata_copy.obs["leiden"].values,
            metric="cosine",
        )
        Sil_leiden_all_hvg = silhouette_score(
            adata_copy.X[:, adata_copy.var["highly_variable"].values],
            adata_copy.obs["leiden"].values,
            metric="cosine",
        )
        Sil_leiden_all_base = silhouette_score(
            adata_copy.X, adata_copy.obs["leiden"].values, metric="cosine"
        )

        if cell_types is not None:
            Sil_leiden_PCA_random = silhouette_score(
                adata_copy.obsm["X_pca"], cell_types_random, metric="cosine"
            )
            Sil_leiden_all_hvg_random = silhouette_score(
                adata_copy.X[:, adata_copy.var["highly_variable"].values],
                cell_types_random,
                metric="cosine",
            )
            Sil_leiden_all_base_random = silhouette_score(
                adata_copy.X, cell_types_random, metric="cosine"
            )
        else:
            (
                Sil_leiden_PCA_random,
                Sil_leiden_all_hvg_random,
                Sil_leiden_all_base_random,
            ) = (None, None, None)

        if cell_types is not None:
            DavBo_bench_PCA = davies_bouldin_score(
                adata_copy.obsm["X_pca"], cell_types
            )
            DavBo_bench_all_hvg = davies_bouldin_score(
                adata_copy.X[:, adata_copy.var["highly_variable"].values],
                cell_types,
            )
            DavBo_bench_all_base = davies_bouldin_score(
                adata_copy.X, cell_types
            )

            DavBo_bench_PCA_random = davies_bouldin_score(
                adata_copy.obsm["X_pca"], cell_types_random
            )
            DavBo_bench_all_hvg_random = davies_bouldin_score(
                adata_copy.X[:, adata_copy.var["highly_variable"].values],
                cell_types_random,
            )
            DavBo_bench_all_base_random = davies_bouldin_score(
                adata_copy.X, cell_types_random
            )
        else:
            DavBo_bench_PCA, DavBo_bench_all_hvg, DavBo_bench_all_base = (
                None,
                None,
                None,
            )
            (
                DavBo_bench_PCA_random,
                DavBo_bench_all_hvg_random,
                DavBo_bench_all_base_random,
            ) = (None, None, None)

        DavBo_leiden_PCA = davies_bouldin_score(
            adata_copy.obsm["X_pca"], adata_copy.obs["leiden"].values
        )
        DavBo_leiden_all_hvg = davies_bouldin_score(
            adata_copy.X[:, adata_copy.var["highly_variable"].values],
            adata_copy.obs["leiden"].values,
        )
        DavBo_leiden_all_base = davies_bouldin_score(
            adata_copy.X, adata_copy.obs["leiden"].values
        )

        if cell_types is not None:
            DavBo_leiden_PCA_random = davies_bouldin_score(
                adata_copy.obsm["X_pca"], cell_types_random
            )
            DavBo_leiden_all_hvg_random = davies_bouldin_score(
                adata_copy.X[:, adata_copy.var["highly_variable"].values],
                cell_types_random,
            )
            DavBo_leiden_all_base_random = davies_bouldin_score(
                adata_copy.X, cell_types_random
            )
        else:
            (
                DavBo_leiden_PCA_random,
                DavBo_leiden_all_hvg_random,
                DavBo_leiden_all_base_random,
            ) = (None, None, None)

        df_score.loc["ARI", method], df_score.loc["ARI_random", method] = (
            ARI,
            ARI_random,
        )
        df_score.loc["NMI", method], df_score.loc["NMI_random", method] = (
            NMIs,
            NMI_random,
        )

        df_score.loc["Sil_bench_UMAP", method] = Sil_bench_UMAP
        df_score.loc["Sil_bench_PCA", method] = Sil_bench_PCA
        df_score.loc["Sil_bench_all_hvg", method] = Sil_bench_all_hvg
        df_score.loc["Sil_bench_all_base", method] = Sil_bench_all_base
        df_score.loc["Sil_bench_PCA_random", method] = Sil_bench_PCA_random
        df_score.loc[
            "Sil_bench_all_hvg_random", method
        ] = Sil_bench_all_hvg_random
        df_score.loc[
            "Sil_bench_all_base_random", method
        ] = Sil_bench_all_base_random

        df_score.loc["Sil_leiden_UMAP", method] = Sil_leiden_UMAP
        df_score.loc["Sil_leiden_PCA", method] = Sil_leiden_PCA
        df_score.loc["Sil_leiden_all_hvg", method] = Sil_leiden_all_hvg
        df_score.loc["Sil_leiden_all_base", method] = Sil_leiden_all_base
        df_score.loc["Sil_leiden_PCA_random", method] = Sil_leiden_PCA_random
        df_score.loc[
            "Sil_leiden_all_hvg_random", method
        ] = Sil_leiden_all_hvg_random
        df_score.loc[
            "Sil_leiden_all_base_random", method
        ] = Sil_leiden_all_base_random

        df_score.loc["DavBo_bench_PCA", method] = DavBo_bench_PCA
        df_score.loc["DavBo_bench_all_hvg", method] = DavBo_bench_all_hvg
        df_score.loc["DavBo_bench_all_base", method] = DavBo_bench_all_base
        df_score.loc["DavBo_bench_PCA_random", method] = DavBo_bench_PCA_random
        df_score.loc[
            "DavBo_bench_all_hvg_random", method
        ] = DavBo_bench_all_hvg_random
        df_score.loc[
            "DavBo_bench_all_base_random", method
        ] = DavBo_bench_all_base_random

        df_score.loc["DavBo_leiden_PCA", method] = DavBo_leiden_PCA
        df_score.loc["DavBo_leiden_all_hvg", method] = DavBo_leiden_all_hvg
        df_score.loc["DavBo_leiden_all_base", method] = DavBo_leiden_all_base
        df_score.loc[
            "DavBo_leiden_PCA_random", method
        ] = DavBo_leiden_PCA_random
        df_score.loc[
            "DavBo_leiden_all_hvg_random", method
        ] = DavBo_leiden_all_hvg_random
        df_score.loc[
            "DavBo_leiden_all_base_random", method
        ] = DavBo_leiden_all_base_random
    #         print(df_score)

    del adata_copy
    gc.collect()
    df_score.to_csv(f"{outdir}/{file_root}_comparison-scores_seed-{seed}.csv")


def friedman_test(arr):
    N, k = arr.shape

    chi = (
        12
        * N
        / (k * (k + 1))
        * (np.sum(arr.mean(0) ** 2) - k * (k + 1) ** 2 / 4)
    )
    F = (N - 1) * chi / (N * (k - 1) - chi)

    df1, df2 = k - 1, (k - 1) * (N - 1)

    return F, f(df1, df2).sf(F)


def argsort_line(increasing, list_vals):
    if increasing == -1:
        return list_vals
    elif increasing:
        return (
            np.argsort(np.argsort(list_vals)) + 1
        )  # sort increasing (lower is best
    else:
        return (
            np.argsort(np.argsort(list_vals)[::-1]) + 1
        )  # sort decreasing (higher is best)


def get_ranking_stats(
    dir_comparisons,
    list_files,
    method_selection,
    increasing=True,
    mode="normal",
):
    dict_df = {}

    for file in list_files:
        df = pd.read_csv(f"{dir_comparisons}/{file}", index_col=0)
        columns = df.columns

        if mode == "normal":
            list_vals = df.loc[method_selection, :].values.tolist()
            dict_df[file] = argsort_line(increasing, list_vals)
        else:
            for i in range(len(df)):
                list_vals = df.iloc[i, :].values.tolist()
                dict_df[file] = argsort_line(increasing, list_vals)

    df_ranks = pd.DataFrame.from_dict(dict_df, orient="index", columns=columns)

    F, pval = friedman_test(df_ranks.values)

    # Posthoc tests
    df_posthoc = posthoc_quade(df_ranks.values)  # First use array
    df_posthoc = df_posthoc.set_index(columns)  # Then set columns and rows
    df_posthoc.columns = columns
    df_posthoc[
        df_posthoc == -1
    ] = 1  # Identical elements in comparison matrix are -1 instead of 1

    return df_ranks, F, pval, df_posthoc


def plot_lab_org_comparison_scores(
    lab,
    org="-",
    read_dir="",
    variables=[],
    increasing=False,
    list_files=None,
    alpha=0.05,
    figsize=(16, 4),
    title="",
    filename="",
    mode="normal",
    lognames=None,  # None: nothing, 0: selects names without log, 1: selects names with log
):

    if isinstance(variables, str):
        variables = [variables]

    if list_files is None:
        list_files = sorted(
            [
                i
                for i in os.listdir(read_dir)
                if lab in i and org in i and "comparison-scores" in i
            ]
        )

    if lognames is not None:
        if lognames:
            list_files = [i for i in list_files if "-log" in i]
        else:
            list_files = [i for i in list_files if "-log" not in i]

    # For the plot on the left (test + post-hoc test)
    df_ranks, F, pval, df_posthoc = get_ranking_stats(
        read_dir, list_files, variables[0], increasing=increasing, mode=mode
    )
    df_ranks_means = df_ranks.mean(0).sort_values()

    methods = df_ranks.columns.tolist()
    columns_sorted = df_ranks_means.index.values

    df_posthoc = df_posthoc.loc[columns_sorted, columns_sorted]

    list_start_end = []
    for idx, col in enumerate(columns_sorted):
        idx_nonsignificant = np.argwhere(
            df_posthoc.loc[col, :].values > alpha
        ).ravel()
        tuple_idx = (idx_nonsignificant[0], idx_nonsignificant[-1])

        if tuple_idx[0] != tuple_idx[1]:
            if (
                len(list_start_end) == 0
            ):  # This part is to remove elements that are inside other elements, and take the biggest one.
                list_start_end.append(tuple_idx)
            else:
                if (tuple_idx[0] >= list_start_end[-1][0]) & (
                    tuple_idx[1] <= list_start_end[-1][1]
                ):
                    pass
                elif (tuple_idx[0] <= list_start_end[-1][0]) & (
                    tuple_idx[1] >= list_start_end[-1][1]
                ):
                    list_start_end[-1] = tuple_idx
                else:
                    list_start_end.append(tuple_idx)

    markers = ["o", "v", "s", "1"]
    basepalette = [
        "#E73F74",
        "#7F3C8D",
        "#11A579",
        "#3969AC",
        "#F2B701",
        "#80BA5A",
        "#E68310",
    ]
    palette = basepalette[: len(methods) - 2] + [
        "#a0a0a0",
        "#505050",
    ]
    dict_palette = dict(zip(methods, palette[: len(methods)]))

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 8]}
    )

    list_libpreps = list(
        dict.fromkeys(
            [i.split("_")[1] + " " + i.split("_")[2] for i in list_files]
        )
    )

    if mode == "normal":
        for libprep_idx, libprep in enumerate(list_libpreps):
            if "log" in libprep:
                files_seeds = [
                    i
                    for i in list_files
                    if ("_" + libprep.replace(" ", "_") in i) & ("log" in i)
                ]
            else:
                files_seeds = [
                    i
                    for i in list_files
                    if ("_" + libprep.replace(" ", "_") in i)
                    & ("log" not in i)
                ]

            for method_idx, method in enumerate(methods):
                axl.plot(
                    [0, len(list_start_end) + 1],
                    [df_ranks.mean(0)[method], df_ranks.mean(0)[method]],
                    c=dict_palette[method],
                )

                for variable_idx, variable in enumerate(variables):
                    list_y = []

                    for seed in range(len(files_seeds)):
                        file = [i for i in files_seeds if f"seed-{seed}" in i][
                            0
                        ]
                        df = pd.read_csv(read_dir + "/" + file, index_col=0)
                        list_y.append(df.loc[variable, method])

                    axr.bar(
                        libprep_idx + (method_idx - len(methods) // 2) * 0.09,
                        np.mean(list_y),
                        width=0.09,
                        yerr=np.std(list_y),
                        color=dict_palette[method],
                    )
    else:
        for libprep_idx, libprep in enumerate(list_files):
            for method_idx, method in enumerate(methods):
                axl.plot(
                    [0, len(list_start_end) + 1],
                    [df_ranks.mean(0)[method], df_ranks.mean(0)[method]],
                    c=dict_palette[method],
                )

                for variable_idx, variable in enumerate(variables):
                    df = pd.read_csv(read_dir + "/" + libprep, index_col=0)
                    list_y = df.loc[:, method]

                    axr.bar(
                        libprep_idx + (method_idx - len(methods) // 2) * 0.09,
                        np.mean(list_y),
                        width=0.09,
                        yerr=np.std(list_y),
                        color=dict_palette[method],
                    )

    for idx, tuple_list in enumerate(list_start_end):
        axl.plot(
            [idx + 1, idx + 1],
            [
                df_ranks_means.iloc[tuple_list[0]] - 0.03,
                df_ranks_means.iloc[tuple_list[1]] + 0.03,
            ],
            c="#808080",
            linewidth=5,
        )

    # Axis formatting
    axr.set_xticks(np.arange(len(list_libpreps)))
    axr.set_xticklabels(list_libpreps, rotation=45, ha="right")
    axr.spines["right"].set_visible(False)
    axr.spines["top"].set_visible(False)

    dict_ylabel = {
        -1: "Values",
        1: "Rank (lower is better)",
        0: "Rank (lower is better)",
    }
    axl.set_ylabel(dict_ylabel[increasing])
    if increasing != -1:
        axl.invert_yaxis()
    axl.set_xticks([])
    axl.spines["right"].set_visible(False)
    axl.spines["top"].set_visible(False)
    axl.spines["bottom"].set_visible(False)

    l1 = axr.legend(
        bbox_to_anchor=(1, 0.75),
        handles=[
            Line2D(
                [0], [0], marker="o", color=palette[method_idx], label=method
            )
            for method_idx, method in enumerate(methods)
        ],
    )
    axr.add_artist(l1)

    if len(variables) > 1:
        l2 = axr.legend(
            handles=[
                Line2D([0], [0], marker=markers[variable_idx], label=variable)
                for variable_idx, variable in enumerate(variables)
            ]
        )
        axr.add_artist(l2)

    plt.title(title)
    for fmt in ["png", "pdf"]:
        fig.savefig(
            f"{os.getcwd()}/figures/comparison_figs/{fmt}/{filename}.{fmt}",
            bbox_inches="tight",
        )


def compare_rankings(
    list_files_1,
    list_files_2,
    read_dir,
    title="",
    title1="",
    title2="",
    variables=[],
    increasing=False,
    alpha=0.1,
    figsize=(16, 4),
    filename="",
    mode="normal",
):

    # For the plot on the left (test + post-hoc test) [NONLOG]
    df_ranks_1, F_1, pval_1, df_posthoc_1 = get_ranking_stats(
        read_dir, list_files_1, variables[0], increasing=increasing, mode=mode
    )
    df_ranks_means_1 = df_ranks_1.mean(0).sort_values()

    methods = df_ranks_1.columns.tolist()
    columns_sorted_1 = df_ranks_means_1.index.values
    df_posthoc_1 = df_posthoc_1.loc[columns_sorted_1, columns_sorted_1]

    # For the plot on the right (test + post-hoc test) [LOG]
    df_ranks_2, F_2, pval_2, df_posthoc_2 = get_ranking_stats(
        read_dir, list_files_2, variables[0], increasing=increasing, mode=mode
    )
    df_ranks_means_2 = df_ranks_2.mean(0).sort_values()

    columns_sorted_2 = df_ranks_means_2.index.values
    df_posthoc_2 = df_posthoc_2.loc[columns_sorted_2, columns_sorted_2]

    list_start_end_1, list_start_end_2 = [], []
    for idx, col in enumerate(columns_sorted_1):
        idx_nonsignificant = np.argwhere(
            df_posthoc_1.loc[col, :].values > alpha
        ).ravel()
        tuple_idx = (idx_nonsignificant[0], idx_nonsignificant[-1])

        if tuple_idx[0] != tuple_idx[1]:
            if (
                len(list_start_end_1) == 0
            ):  # This part is to remove elements that are inside other elements, and take the biggest one.
                list_start_end_1.append(tuple_idx)
            else:
                if (tuple_idx[0] >= list_start_end_1[-1][0]) & (
                    tuple_idx[1] <= list_start_end_1[-1][1]
                ):
                    pass
                elif (tuple_idx[0] <= list_start_end_1[-1][0]) & (
                    tuple_idx[1] >= list_start_end_1[-1][1]
                ):
                    list_start_end_1[-1] = tuple_idx
                else:
                    list_start_end_1.append(tuple_idx)

    for idx, col in enumerate(columns_sorted_2):
        idx_nonsignificant = np.argwhere(
            df_posthoc_2.loc[col, :].values > alpha
        ).ravel()
        tuple_idx = (idx_nonsignificant[0], idx_nonsignificant[-1])

        if tuple_idx[0] != tuple_idx[1]:
            if (
                len(list_start_end_2) == 0
            ):  # This part is to remove elements that are inside other elements, and take the biggest one.
                list_start_end_2.append(tuple_idx)
            else:
                if (tuple_idx[0] >= list_start_end_2[-1][0]) & (
                    tuple_idx[1] <= list_start_end_2[-1][1]
                ):
                    pass
                elif (tuple_idx[0] <= list_start_end_2[-1][0]) & (
                    tuple_idx[1] >= list_start_end_2[-1][1]
                ):
                    list_start_end_2[-1] = tuple_idx
                else:
                    list_start_end_2.append(tuple_idx)

    basepalette = [
        "#E73F74",
        "#7F3C8D",
        "#11A579",
        "#3969AC",
        "#F2B701",
        "#80BA5A",
        "#E68310",
    ]
    palette = basepalette[: len(methods) - 2] + [
        "#a0a0a0",
        "#505050",
    ]
    dict_palette = dict(zip(methods, palette[: len(methods)]))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=figsize)

    for method_idx, method in enumerate(methods):
        axl.plot(
            [0, len(list_start_end_1) + 1],
            [df_ranks_1.mean(0)[method], df_ranks_1.mean(0)[method]],
            c=dict_palette[method],
        )

        axr.plot(
            [0, len(list_start_end_2) + 1],
            [df_ranks_2.mean(0)[method], df_ranks_2.mean(0)[method]],
            c=dict_palette[method],
        )

    for idx, tuple_list in enumerate(list_start_end_1):
        axl.plot(
            [idx + 1, idx + 1],
            [
                df_ranks_means_1.iloc[tuple_list[0]] - 0.03,
                df_ranks_means_1.iloc[tuple_list[1]] + 0.03,
            ],
            c="#808080",
            linewidth=5,
        )

    for idx, tuple_list in enumerate(list_start_end_2):
        axr.plot(
            [idx + 1, idx + 1],
            [
                df_ranks_means_2.iloc[tuple_list[0]] - 0.03,
                df_ranks_means_2.iloc[tuple_list[1]] + 0.03,
            ],
            c="#808080",
            linewidth=5,
        )

    # Axis formatting
    for ax in [axl, axr]:
        ax.set_ylabel("Rank (lower is better)")
        ax.set_xticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_ylim(
            min([axl.get_ylim()[0], axr.get_ylim()[0]]),
            max([axl.get_ylim()[1], axr.get_ylim()[1]]),
        )

    for ax in [axl, axr]:
        ax.invert_yaxis()

    l1 = axr.legend(
        bbox_to_anchor=(1, 0.75),
        handles=[
            Line2D(
                [0], [0], marker="o", color=palette[method_idx], label=method
            )
            for method_idx, method in enumerate(methods)
        ],
    )
    ax.add_artist(l1)

    axl.set_title(title1)
    axr.set_title(title2)
    plt.suptitle(title, y=1.03)

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(
            f"{os.getcwd()}/figures/comparison_figs/{fmt}/{filename}.{fmt}",
            bbox_inches="tight",
        )


def compare_values(
    list_files_1,
    list_files_2,
    read_dir,
    title="",
    title1="",
    title2="",
    variables=[],
    increasing=False,
    alpha=0.1,
    figsize=(16, 4),
    filename="",
    mode="normal",
):

    list_diffs, list_methods = [], []

    for file_idx in range(len(list_files_1)):
        df_1, df_2 = (
            pd.read_csv(read_dir + list_files_1[file_idx], index_col=0),
            pd.read_csv(read_dir + list_files_2[file_idx], index_col=0),
        )
        list_diffs += (
            df_1.loc[variables[0], :] - df_2.loc[variables[0], :]
        ).tolist()
        list_methods += df_1.columns.tolist()

    methods = df_1.columns
    basepalette = [
        "#E73F74",
        "#7F3C8D",
        "#11A579",
        "#3969AC",
        "#F2B701",
        "#80BA5A",
        "#E68310",
    ]
    palette = basepalette[: len(methods) - 2] + [
        "#a0a0a0",
        "#505050",
    ]
    #     dict_palette = dict(zip(methods, palette[: len(methods)]))

    df_val_diffs = pd.DataFrame({"diff": list_diffs, "method": list_methods})

    fig, ax = plt.subplots(figsize=figsize)
    sns.swarmplot(
        x="method", y="diff", data=df_val_diffs, palette=palette, ax=ax, s=4
    )
    fig.suptitle(title)

    for method_idx, method in enumerate(methods):
        values_method = df_val_diffs["diff"][
            df_val_diffs["method"] == method
        ].values
        if np.sum(values_method) == 0:
            p = 0.5
        else:
            t, p = wilcoxon(values_method)

        if (
            np.median(values_method) >= 0
        ):  # THIS IS BECAUSE WE WAN'T A SINGLE-TAILED TEST WITH MU > 0!!!!
            p = 1 / 2 * p
        else:
            p = 1 - 1 / 2 * p

        if np.isnan(p):
            p = 0.5

        if p < 0.01:
            pstr = f"{p:.3e}"
        elif p > 0.9:
            pstr = "~1"
        else:
            pstr = f"{p:.2f}"

        xmov = -0.5 if p < 0.01 else -0.2

        ax.text(method_idx + xmov, 0.1 + max(values_method), f"{pstr}")

    ax.plot([-0.5, len(methods) - 0.2], [0, 0], c="#bcbcbc")

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.15)
    ax.set_xlim(-0.8, len(methods) + 0.1)

    for fmt in ["png", "pdf"]:
        fig.savefig(
            f"{os.getcwd()}/figures/comparison_figs/{fmt}/{filename}.{fmt}",
            bbox_inches="tight",
        )


def create_UMAP_adataset_libprep_org(
    adata_dir, df_rank_dir, lib_prep, org, lab, log
):
    list_methods = [
        "triku",
        "m3drop",
        "nbumi",
        "scanpy",
        "brennecke",
        "scry",
        "std",
    ]

    adata = sc.read_h5ad(f"{adata_dir}/{lib_prep}_{org}.h5ad")
    df_rank = pd.read_csv(
        f"{df_rank_dir}/{lab}_{lib_prep}_{org}_feature_ranks.csv", index_col=0
    )

    if log:
        sc.pp.log1p(adata)

    tk.tl.triku(
        adata, n_procs=1,
    )
    n_HVG = np.sum(adata.var["highly_variable"].values)
    print("n_HVG", n_HVG)
    col_cell_types = "cell_types" if "cell_types" in adata.obs else "CellType"
    cell_types = adata.obs[col_cell_types]

    dict_return = {}

    dict_other_stuff = {
        "cell_types": cell_types.values,
        "mean": adata.X.mean(0).ravel(),
        "std": adata.X.std(0).ravel(),
        "per_0": (adata.X == 0).mean(0).ravel(),
        "CV2": (adata.X.mean(0).ravel() / adata.X.std(0).ravel()) ** 2,
    }
    for method in list_methods:
        adata_copy = adata.copy()
        apply_log = not log

        if method == "scanpy":
            sc.pp.log1p(
                adata_copy
            )  # We need scanpy to calculate the dispersions, so we take advantage of log being already calculated
            apply_log = False
            ret = sc.pp.highly_variable_genes(
                adata_copy, n_top_genes=n_HVG, inplace=False
            )
            dict_other_stuff["disp"], dict_other_stuff["disp_norm"] = (
                ret["dispersions"],
                ret["dispersions_norm"],
            )

        if method != "triku":
            adata_copy.var["highly_variable"] = [
                i in df_rank[method].sort_values().index[:n_HVG]
                for i in adata_copy.var_names
            ]

        leiden_sol, res_sol = clustering_binary_search(
            adata_copy,
            min_res=0.1,
            max_res=2,
            max_depth=7,
            seed=0,
            n_target_c=len(set(cell_types)),
            features=adata_copy[
                :, adata_copy.var["highly_variable"] == True  # noqa
            ].var_names,
            apply_log=apply_log,
            transform_adata=True,
        )

        sc.tl.umap(adata_copy)

        dict_return[method] = {
            "UMAP": adata_copy.obsm["X_umap"],
            "leiden": leiden_sol.values,
            "highly_variable": adata_copy.var["highly_variable"].values,
        }

        del adata_copy
        gc.collect()

    dict_return["other_stuff"] = dict_other_stuff

    return f"{lib_prep} {org}", dict_return


def create_dict_UMAPs_datasets(
    adata_dir,
    df_rank_dir,
    lab,
    lib_preps,
    list_orgs=["human", "mouse"],
    log=False,
):
    list_org_preps_all = list(product(*[lib_preps, list_orgs]))
    list_org_preps_exist = []

    for lib_prep, org in list_org_preps_all:
        for file in os.listdir(adata_dir):
            if org in file and lib_prep in file:
                list_org_preps_exist.append((lib_prep, org))
                break

    create_UMAP_adataset_libprep_org_remote = ray.remote(
        create_UMAP_adataset_libprep_org
    )

    ray.init(ignore_reinit_error=True)

    list_ids = [
        create_UMAP_adataset_libprep_org_remote.remote(
            adata_dir, df_rank_dir, lib_prep, org, lab, log
        )
        for lib_prep, org in list_org_preps_exist
    ]
    list_returns = ray.get(list_ids)

    ray.shutdown()

    return dict(list_returns)


def plot_UMAPs_datasets(dict_returns, fig_save_dir, lab, figsize=(25, 40)):
    list_rows = list(dict_returns.keys())
    list_methods = list(dict_returns[list_rows[0]].keys())[:-1]

    fig_leiden, axs_leiden = plt.subplots(
        len(list_rows), len(list_methods), figsize=figsize
    )
    fig_cell_types, axs_cell_types = plt.subplots(
        len(list_rows), len(list_methods), figsize=figsize
    )

    for row_idx, row_name in enumerate(list_rows):
        for col_idx, col_name in enumerate(list_methods):
            UMAP_coords = dict_returns[row_name][col_name]["UMAP"]
            leiden_labels = dict_returns[row_name][col_name]["leiden"]
            cell_types = dict_returns[row_name]["other_stuff"]["cell_types"]
            # Names are too long for plots, so we are goinf to simplify them
            set_cell_types = list(dict.fromkeys(cell_types))
            cell_types = pd.Categorical(
                [f"C{set_cell_types.index(i)}" for i in cell_types]
            )

            # We will create the adata to plot the labels, its much easier than programming the feature by yourself.
            adata = sc.AnnData(X=np.zeros((UMAP_coords.shape[0], 100)))
            (
                adata.obsm["X_umap"],
                adata.obs["leiden"],
                adata.obs["cell_type"],
            ) = (
                UMAP_coords,
                leiden_labels,
                cell_types,
            )

            sc.pl.umap(
                adata,
                color="leiden",
                ax=axs_leiden[row_idx][col_idx],
                legend_loc="on data",
                show=False,
                s=35,
                legend_fontweight=1000,
            )
            sc.pl.umap(
                adata,
                color="cell_type",
                ax=axs_cell_types[row_idx][col_idx],
                legend_loc="on data",
                show=False,
                s=35,
                legend_fontweight=1000,
            )

            for axs in [axs_leiden, axs_cell_types]:
                axs[row_idx][col_idx].set_xlabel("")
                axs[row_idx][col_idx].set_ylabel("")
                axs[row_idx][col_idx].set_title("")

                if row_idx == 0:
                    axs[row_idx][col_idx].set_xlabel(f"{col_name}")
                    axs[row_idx][col_idx].xaxis.set_label_position("top")

                if col_idx == 0:
                    axs[row_idx][col_idx].set_ylabel(f"{row_name}")
                    axs[row_idx][col_idx].yaxis.set_label_position("left")

    plt.tight_layout()
    fig_leiden.savefig(
        f"{fig_save_dir}/pdf/{lab}_UMAP_leiden.pdf", bbox_inches="tight"
    )
    fig_cell_types.savefig(
        f"{fig_save_dir}/pdf/{lab}_UMAP_cell_types.pdf", bbox_inches="tight"
    )
    fig_leiden.savefig(
        f"{fig_save_dir}/png/{lab}_UMAP_leiden.png",
        bbox_inches="tight",
        dpi=400,
    )
    fig_cell_types.savefig(
        f"{fig_save_dir}/png/{lab}_UMAP_cell_types.png",
        bbox_inches="tight",
        dpi=400,
    )


#     for fmt in ['png', 'pdf']:
#         os.makedirs(f'{fig_save_dir}/{fmt}', exist_ok=True)
#         fig_leiden.savefig(f'{fig_save_dir}/{fmt}/{lab}_UMAP_leiden.{fmt}', bbox_inches='tight')
#         fig_cell_types.savefig(f'{fig_save_dir}/{fmt}/{lab}_UMAP_cell_types.{fmt}', bbox_inches='tight')


def plot_XY(
    dict_returns,
    x_var,
    y_var,
    fig_save_dir,
    lab,
    figsize=(20, 35),
    logx=True,
    logy=True,
    title="",
):
    list_rows = list(dict_returns.keys())
    list_methods = list(dict_returns[list_rows[0]].keys())[:-1]

    fig, axs = plt.subplots(len(list_rows), len(list_methods), figsize=figsize)

    for row_idx, row_name in enumerate(list_rows):
        for col_idx, col_name in enumerate(list_methods):
            x_coords = dict_returns[row_name]["other_stuff"][x_var]
            y_coords = dict_returns[row_name]["other_stuff"][y_var]
            highly_variable = dict_returns[row_name][col_name][
                "highly_variable"
            ]

            if logx:
                x_coords = np.log10(x_coords)
            if logy:
                y_coords = np.log10(y_coords)

            axs[row_idx][col_idx].scatter(
                x_coords[highly_variable is False],
                y_coords[highly_variable is False],
                c="#cbcbcb",
                alpha=0.05,
                s=2,
            )
            axs[row_idx][col_idx].scatter(
                x_coords[highly_variable is True],
                y_coords[highly_variable is True],
                c="#007ab7",
                alpha=0.2,
                s=2,
            )

            if row_idx == 0:
                axs[row_idx][col_idx].set_title(f"{col_name}")

            if col_idx == 0:
                axs[row_idx][col_idx].set_ylabel(
                    f"{row_name}".replace(" ", "\n")
                )
                axs[row_idx][col_idx].yaxis.set_label_position("left")

    fig.suptitle(title)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        os.makedirs(f"{fig_save_dir}/{fmt}", exist_ok=True)
        fig.savefig(
            f"{fig_save_dir}/{fmt}/{lab}_{x_var}-VS-{y_var}.{fmt}",
            bbox_inches="tight",
        )

    plt.show()
