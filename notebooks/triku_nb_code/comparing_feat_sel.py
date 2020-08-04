import gc
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import scanpy as sc
import scipy.sparse as spr
from matplotlib.lines import Line2D
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
            is True  # Revert to == True if it fails !!!!!!
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


def plot_lab_org_comparison_scores(
    lab,
    org="-",
    read_dir="",
    variables=[],
    figsize=(10, 6),
    title="",
    filename="",
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if isinstance(variables, str):
        variables = [variables]

    markers = ["o", "v", "s", "1"]
    palette = prism

    list_files = [
        i
        for i in os.listdir(read_dir)
        if lab in i and org in i and "comparison-scores" in i
    ]
    list_libpreps = sorted(
        list({i.split("_")[1] + " " + i.split("_")[2] for i in list_files})
    )

    for libprep_idx, libprep in enumerate(list_libpreps):
        files_seeds = [
            i for i in list_files if "_" + libprep.replace(" ", "_") in i
        ]  # This may fail for mereu and Ding now!

        for seed in range(len(files_seeds)):
            file = [i for i in files_seeds if f"seed-{seed}" in i][0]
            df = pd.read_csv(read_dir + "/" + file, index_col=0)

            methods = df.columns.tolist()
            for method_idx, method in enumerate(methods):
                for variable_idx, variable in enumerate(variables):
                    ax.scatter(
                        libprep_idx + (method_idx - len(methods) // 2) * 0.07,
                        df.loc[variable, method],
                        marker=markers[variable_idx],
                        c=palette[method_idx],
                    )

    ax.set_xticks(np.arange(len(list_libpreps)))
    ax.set_xticklabels(list_libpreps, rotation=45, ha="right")

    l1 = ax.legend(
        bbox_to_anchor=(1, 0.75),
        handles=[
            Line2D(
                [0], [0], marker="o", color=palette[method_idx], label=method
            )
            for method_idx, method in enumerate(methods)
        ],
    )
    ax.add_artist(l1)

    if len(variables) > 1:
        l2 = ax.legend(
            handles=[
                Line2D([0], [0], marker=markers[variable_idx], label=variable)
                for variable_idx, variable in enumerate(variables)
            ]
        )
        ax.add_artist(l2)

    plt.title(title)
    for fmt in ["png", "pdf"]:
        fig.savefig(
            f"{os.getcwd()}/figures/comparison_figs/{fmt}/{filename}.{fmt}",
            bbox_inches="tight",
        )


def create_UMAP_adataset_libprep_org(
    adata_dir, df_rank_dir, lib_prep, org, lab
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
        apply_log = True

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
                :, adata_copy.var["highly_variable"] is True
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
    adata_dir, df_rank_dir, lab, lib_preps, list_orgs=["human", "mouse"],
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
            adata_dir, df_rank_dir, lib_prep, org, lab
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
