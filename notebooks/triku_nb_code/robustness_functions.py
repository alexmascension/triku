import gc
import os
from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as sts
from tqdm.notebook import tqdm

import triku as tk
from triku.tl._triku_functions import subtract_median


def run_batch(adata, windows, n_comps, knns, seeds, save_dir, dataset_prefix):
    # We have to run an array of selections. To do that, one of the parameters above must be
    # a list, and the rest a list of one integer. They can also be all lists, but consider that the
    # calculation will take time. Once triku is run, we will export the distances for each combination
    # as csv. Each csv will contain the median-corrected distances, with and without correction with
    # randomization, for a determined combination of window / n_comp / knn and seed (we will use 3 or 5 seeds
    # for replication purposes).

    for window, n_comp, knn, seed in tqdm(
        product(*[windows, n_comps, knns, seeds])
    ):
        print(window, n_comp, knn, seed)
        save_file = "{save_dir}/{pref}-w_{w}-comps_{n_comps}-knn_{knn}-seed_{seed}.csv".format(
            save_dir=save_dir,
            pref=dataset_prefix,
            w=window,
            n_comps=n_comp,
            knn=knn,
            seed=seed,
        )

        if os.path.exists(save_file):
            print("FILE EXISTS!")
        else:
            try:
                tk.tl.triku(
                    adata,
                    n_windows=window,
                    n_comps=n_comp,
                    knn=knn,
                    random_state=seed,
                    verbose="triku",
                    n_procs=25,
                )
                distances_with_random = adata.var["triku_distance"].values
                mean_exp = adata.X.sum(0)
                distances_without_random = subtract_median(
                    x=mean_exp,
                    y=adata.var["triku_distance_uncorrected"].values,
                    n_windows=window,
                )
                print(adata.var["triku_distance_uncorrected"].values[:5])
                df_res = pd.DataFrame(
                    data={
                        "emd_random_correction": distances_with_random,
                        "emd_no_correction": distances_without_random,
                    },
                    index=adata.var_names.values,
                )

            except BaseException:
                df_res = pd.DataFrame(
                    data={
                        "emd_random_correction": [np.NaN]
                        * len(adata.var_names),
                        "emd_no_correction": [np.NaN] * len(adata.var_names),
                    },
                    index=adata.var_names.values,
                )

            df_res.to_csv(save_file)
            del [df_res]
            gc.collect()


def run_all_batches(lib_preps, orgs, dataset, read_dir, save_dir):
    for lib_prep, org in tqdm(product(*[lib_preps, orgs])):
        file_in = None
        for file in os.listdir(read_dir):
            if (
                org in file in file
                and lib_prep in file
                and file.endswith(".h5ad")
            ):
                file_in = file

        if file_in is None:
            print(f"File for {lib_prep} in {org} not found.")
            continue

        print(file_in)
        adata = sc.read_h5ad(read_dir + file_in)
        sqr_n_cells = int(adata.shape[0] ** 0.5)

        run_batch(
            adata,
            windows=[100],
            n_comps=[3, 5, 10, 20, 30, 40, 50, 100],
            knns=[sqr_n_cells + 1],
            seeds=[0, 1, 2, 3, 4],
            save_dir=save_dir,
            dataset_prefix=lib_prep + "_" + dataset + "_" + org,
        )

        run_batch(
            adata,
            windows=[100],
            n_comps=[30],
            knns=[
                sqr_n_cells // 20 + 1,
                sqr_n_cells // 10 + 1,
                sqr_n_cells // 5 + 1,
                sqr_n_cells // 2 + 1,
                sqr_n_cells + 1,
                int(sqr_n_cells * 1.5) + 1,
                sqr_n_cells * 2 + 1,
                sqr_n_cells * 4 + 1,
            ],
            seeds=[0, 1, 2, 3, 4],
            save_dir=save_dir,
            dataset_prefix=lib_prep + "_" + dataset + "_" + org,
        )

        run_batch(
            adata,
            windows=[10, 20, 30, 50, 100, 200, 500, 1000],
            n_comps=[30],
            knns=[sqr_n_cells + 1],
            seeds=[0, 1, 2, 3, 4],
            save_dir=save_dir,
            dataset_prefix=lib_prep + "_" + dataset + "_" + org,
        )

        del adata
        gc.collect()


def return_knn_indices(save_dir, org, lib_prep, dataset):
    knn_list = []
    for file in os.listdir(save_dir):
        if (
            org in file
            and lib_prep in file
            and "w_100-" in file
            and "comps_30-" in file
            and dataset in file
        ):
            knn_str = file[file.find("knn") + 4 :]
            knn_list.append(int(knn_str[: knn_str.find("-")]))
    knn_list = sorted(list(dict.fromkeys(knn_list)))
    return knn_list


def return_pca_indices(save_dir, org, lib_prep, dataset):
    # We need to recover the fixed kNN value. This value is the 5th value on the knn_list; so we will take it.
    knn_pinpoint = return_knn_indices(save_dir, org, lib_prep, dataset)[4]

    # Now we get the list of n_comps values
    pca_list = []
    for file in os.listdir(save_dir):
        if (
            org in file
            and lib_prep in file
            and "w_100-" in file
            and "knn_%s-" % knn_pinpoint in file
            and dataset in file
        ):
            pca_str = file[file.find("comps") + 6 :]
            pca_list.append(int(pca_str[: pca_str.find("-")]))
    pca_list = sorted(list(dict.fromkeys(pca_list)))
    return pca_list, knn_pinpoint


def return_window_indices(save_dir, org, lib_prep, dataset):
    # We need to recover the fixed kNN value. This value is the 5th value on the knn_list; so we will take it.
    knn_pinpoint = return_knn_indices(save_dir, org, lib_prep, dataset)[4]

    # Now we get the list of n_comps values
    w_list = []
    for file in os.listdir(save_dir):
        if (
            org in file
            and lib_prep in file
            and "comps_30-" in file
            and "knn_%s-" % knn_pinpoint in file
            and dataset in file
        ):
            w_str = file[file.find("w_") + 2 :]
            w_list.append(int(w_str[: w_str.find("-")]))
    w_list = sorted(list(dict.fromkeys(w_list)))
    return w_list, knn_pinpoint


def return_relative_noise(df_1, df_2, select_index_df):
    relative_noise_non_rand = list(
        (
            df_1["emd_no_correction"].loc[select_index_df].values
            - df_2["emd_no_correction"].loc[select_index_df].values
        )
        / (
            np.abs(df_1["emd_no_correction"].loc[select_index_df].values)
            + np.abs(df_2["emd_no_correction"].loc[select_index_df].values)
        )
    )
    relative_noise_rand = list(
        (
            df_1["emd_random_correction"].loc[select_index_df].values
            - df_2["emd_random_correction"].loc[select_index_df].values
        )
        / (
            np.abs(df_1["emd_random_correction"].loc[select_index_df].values)
            + np.abs(df_2["emd_random_correction"].loc[select_index_df].values)
        )
    )
    return relative_noise_rand, relative_noise_non_rand


def return_percentage_overlap(df_1, df_2, min_n_feats, max_n_feats):
    feats_1_no_cor = (
        df_1.sort_values(by="emd_no_correction", ascending=False)
        .index[min_n_feats:max_n_feats]
        .values
    )
    feats_2_no_cor = (
        df_2.sort_values(by="emd_no_correction", ascending=False)
        .index[min_n_feats:max_n_feats]
        .values
    )
    feats_1_rand = (
        df_1.sort_values(by="emd_random_correction", ascending=False)
        .index[min_n_feats:max_n_feats]
        .values
    )
    feats_2_rand = (
        df_2.sort_values(by="emd_random_correction", ascending=False)
        .index[min_n_feats:max_n_feats]
        .values
    )

    percentage_overlap_non_rand = len(
        np.intersect1d(feats_1_no_cor, feats_2_no_cor)
    ) / (max_n_feats - min_n_feats)
    percentage_overlap_rand = len(
        np.intersect1d(feats_1_rand, feats_2_rand)
    ) / (max_n_feats - min_n_feats)

    return [percentage_overlap_rand], [percentage_overlap_non_rand]


def return_correlation(df_1, df_2, min_n_feats, max_n_feats):
    feats_1_no_cor = (
        df_1["emd_no_correction"]
        .sort_values(ascending=False)
        .iloc[min_n_feats:max_n_feats]
        .values
    )
    feats_2_no_cor = (
        df_2["emd_no_correction"]
        .sort_values(ascending=False)
        .iloc[min_n_feats:max_n_feats]
        .values
    )
    feats_1_rand = (
        df_1["emd_random_correction"]
        .sort_values(ascending=False)
        .iloc[min_n_feats:max_n_feats]
        .values
    )
    feats_2_rand = (
        df_2["emd_random_correction"]
        .sort_values(ascending=False)
        .iloc[min_n_feats:max_n_feats]
        .values
    )

    for array in [feats_1_no_cor, feats_2_no_cor, feats_1_rand, feats_2_rand]:
        np.nan_to_num(array, copy=False)

    correlation_non_rand = sts.pearsonr(feats_1_no_cor, feats_2_no_cor)
    correlation_rand = sts.pearsonr(feats_1_rand, feats_2_rand)

    return [correlation_non_rand[0]], [correlation_rand[0]]


def random_noise_parameter(
    lib_prep, org, dataset, save_dir, min_n_feats, max_n_feats, what, by
):
    list_dists_non_randomized, list_dists_randomized, list_param_value = (
        [],
        [],
        [],
    )

    knn_list = return_knn_indices(save_dir, org, lib_prep, dataset)
    pca_list = return_pca_indices(save_dir, org, lib_prep, dataset)
    if by == "knn":
        parameter_list = knn_list
    elif by == "pca":
        parameter_list, _ = pca_list

    for val in parameter_list:
        list_dfs = []
        for file in os.listdir(save_dir):
            if by == "knn":
                static_comp = (
                    "w_100-" in file
                    and "comps_30-" in file
                    and dataset in file
                )
                dyn_comp = "knn_" + str(val) in file
            elif by == "pca":
                static_comp = (
                    "w_100-" in file
                    and "knn_" + str(knn_list[4]) + "-" in file
                    and dataset in file
                )
                dyn_comp = "comps_{}-".format(val) in file

            if org in file and lib_prep in file and static_comp and dyn_comp:
                df = pd.read_csv(save_dir + file)
                df = df.set_index("Unnamed: 0")
                list_dfs.append(df)

        # find the genes with biggest distance. We will only choose the last dataframe, but for other
        # stuff we will do a combination of all of them
        select_index_df = (
            (df["emd_no_correction"] + df["emd_random_correction"])
            .sort_values(ascending=False)
            .index[min_n_feats:max_n_feats]
        )

        for i in range(len(list_dfs)):
            for j in range(len(list_dfs)):
                if i > j:
                    df_1, df_2 = list_dfs[i], list_dfs[j]
                    if what == "relative noise":
                        what_rand, what_non_rand = return_relative_noise(
                            df_1, df_2, select_index_df
                        )
                    elif what == "overlap":
                        what_rand, what_non_rand = return_percentage_overlap(
                            df_1, df_2, min_n_feats, max_n_feats
                        )
                    else:
                        what_rand, what_non_rand = None, None

                    list_dists_non_randomized += what_non_rand
                    list_dists_randomized += what_rand

                    list_param_value += [val] * len(what_non_rand)

    df_violin = pd.DataFrame(
        {
            "d": np.abs(list_dists_non_randomized + list_dists_randomized),
            by: list_param_value * 2,
            "randomized": ["No"] * len(list_dists_non_randomized)
            + ["Yes"] * len(list_dists_randomized),
        }
    )
    return df_violin


def compare_parameter(
    lib_prep, org, dataset, save_dir, min_n_feats, max_n_feats, what, by
):
    list_dists_non_randomized, list_dists_randomized, list_knn = [], [], []

    knn_list = return_knn_indices(save_dir, org, lib_prep, dataset)
    pca_list = return_pca_indices(save_dir, org, lib_prep, dataset)
    window_list = return_window_indices(save_dir, org, lib_prep, dataset)

    # We first fill on list of dfs with knn = sqrt(N)
    list_dfs_knn_1 = []
    for file in os.listdir(save_dir):
        if (
            org in file
            and lib_prep in file
            and "w_100-" in file
            and "comps_30-" in file
            and "knn_" + str(knn_list[4]) in file
            and dataset in file
        ):
            df = pd.read_csv(save_dir + file)
            df = df.set_index("Unnamed: 0")
            list_dfs_knn_1.append(df)

    if by == "knn":
        parameter_list = knn_list
    elif by == "pca":
        parameter_list, _ = pca_list
    elif by == "w":
        parameter_list, _ = window_list

    for val in parameter_list:
        list_dfs_knn_2 = []
        for file in os.listdir(save_dir):

            if by == "knn":
                static_comp = (
                    "w_100-" in file
                    and "comps_30-" in file
                    and dataset in file
                )
                dyn_comp = "knn_" + str(val) in file
            elif by == "pca":
                static_comp = (
                    "w_100-" in file
                    and "knn_" + str(knn_list[4]) + "-" in file
                    and dataset in file
                )
                dyn_comp = "comps_{}-".format(val) in file
            elif by == "w":
                static_comp = (
                    "comps_30-" in file
                    and "knn_" + str(knn_list[4]) + "-" in file
                    and dataset in file
                )
                dyn_comp = "w_{}-".format(val) in file

            if org in file and lib_prep in file and static_comp and dyn_comp:
                df = pd.read_csv(save_dir + file)
                df = df.set_index("Unnamed: 0")
                list_dfs_knn_2.append(df)

        for i in range(len(list_dfs_knn_1)):
            for j in range(len(list_dfs_knn_2)):
                df_1, df_2 = list_dfs_knn_1[i], list_dfs_knn_2[j]
                if what == "overlap":
                    what_rand, what_non_rand = return_percentage_overlap(
                        df_1, df_2, min_n_feats, max_n_feats
                    )
                elif what == "correlation":
                    what_rand, what_non_rand = return_correlation(
                        df_1, df_2, min_n_feats, max_n_feats
                    )
                else:
                    what_rand, what_non_rand = None, None

                list_dists_non_randomized += what_non_rand
                list_dists_randomized += what_rand

                list_knn += [val] * len(what_non_rand)

    df_violin = pd.DataFrame(
        {
            "d": np.abs(list_dists_non_randomized + list_dists_randomized),
            by: list_knn * 2,
            "randomized": ["No"] * len(list_dists_non_randomized)
            + ["Yes"] * len(list_dists_randomized),
        }
    )
    return df_violin


def plot_scatter_parameter(
    list_dfs,
    categories,
    lib_prep,
    org,
    dataset,
    by,
    figsize=(7, 4),
    step=25,
    palette="sunsetcontrast3",
    title="",
    ylabel="",
    save_dir="figures/robustness_figs",
    plot_random=True,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(title.replace("_", " "))
    val_list = sorted(list(dict.fromkeys(list_dfs[0][by].values)))
    # Set params for plot:
    if by == "knn":
        ticks = [
            r"$\sqrt{N}/20$",
            r"$\sqrt{N}/10$",
            r"$\sqrt{N}/5$",
            r"$\sqrt{N}/2$",
            r"$\sqrt{N}$ (%s)" % val_list[4],
            r"$1.5\sqrt{N}$",
            r"$2\sqrt{N}$",
            r"$4\sqrt{N}$",
        ]
        xlabel = "Number of kNN"
    elif by == "pca":
        ticks = val_list
        xlabel = "Number of PCA components"
    elif by == "w":
        ticks = val_list
        xlabel = "Number of windows for median correction"

    for val_idx in range(len(val_list)):
        w, sep, s, alpha = 0.25, 0.05, 2, 0.20
        if palette == "sunsetcontrast3":
            list_colors = ["#fcde9c", "#e34f6f", "#7c1d6f"]
        elif palette == "sunsetmid3":
            list_colors = ["#faa476", "#dc3977", "#7c1d6f"]
        elif palette == "sunsetmid4":
            list_colors = ["#faa476", "#f0746e", "#dc3977", "#7c1d6f"]
        for df_idx, sub_df in enumerate(list_dfs):
            sub_df_ran = sub_df["d"][
                (sub_df[by] == val_list[val_idx])
                & (sub_df["randomized"] == "Yes")
            ].values[::step]
            sub_df_no_ran = sub_df["d"][
                (sub_df[by] == val_list[val_idx])
                & (sub_df["randomized"] == "No")
            ].values[::step]

            alpha_idx = np.round(
                alpha + (1 - alpha) * (1 + df_idx) / (len(list_dfs)), 2
            )  # wild 1.0000000000002 LOL
            if plot_random:
                ax.scatter(
                    val_idx + sep + w * np.random.rand(len(sub_df_ran)),
                    sub_df_ran,
                    c=list_colors[df_idx],
                    s=s,
                    alpha=alpha_idx,
                )
                ax.scatter(
                    val_idx - sep - w * np.random.rand(len(sub_df_no_ran)),
                    sub_df_no_ran,
                    c=list_colors[df_idx],
                    s=s,
                    alpha=alpha_idx,
                )
            else:
                ax.scatter(
                    val_idx - 2 * w * np.random.rand(len(sub_df_no_ran)),
                    sub_df_no_ran,
                    c=list_colors[df_idx],
                    s=s,
                    alpha=alpha_idx,
                )

    plt.xticks(np.arange(len(val_list)), ticks)
    legend_elements = [
        mpl.lines.Line2D(
            [0], [0], marker="o", color=list_colors[j], label=categories[j]
        )
        for j in range(len(list_colors))
    ]
    ax.legend(handles=legend_elements, title="N features")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    os.makedirs(save_dir + "/png", exist_ok=True)
    os.makedirs(save_dir + "/pdf", exist_ok=True)

    plt.savefig(
        save_dir
        + "/pdf/{}_{}_{}_{}.pdf".format(
            title.replace(",", ""), lib_prep, dataset, org
        ),
        format="pdf",
    )
    plt.savefig(
        save_dir
        + "/png/{}_{}_{}_{}.png".format(
            title.replace(",", ""), lib_prep, dataset, org
        ),
        format="png",
        dpi=350,
    )


def plot_scatter_datasets(
    list_dict_dfs,
    org,
    by,
    figsize=(7, 4),
    palette="prism",
    title="",
    ylabel="",
    save_dir="figures/robustness_figs",
):
    # List_dict_dfs will contain as many dicts as categories, and in each dict, with the category name we will
    # include the dataframe with the sample name. In this case sample will refer to the library processsing technique.

    if palette == "prism":
        list_colors = [
            "#5F4690",
            "#1D6996",
            "#38A6A5",
            "#0F8554",
            "#73AF48",
            "#EDAD08",
            "#E17C05",
            "#CC503E",
            "#94346E",
            "#6F4070",
            "#994E95",
            "#666666",
        ]
    elif palette == "bold":
        list_colors = [
            "#7F3C8D",
            "#11A579",
            "#3969AC",
            "#F2B701",
            "#E73F74",
            "#80BA5A",
            "#E68310",
            "#008695",
            "#CF1C90",
            "#f97b72",
            "#4b4b8f",
            "#A5AA99",
        ]
    elif palette == "vivid":
        list_colors = [
            "#E58606",
            "#5D69B1",
            "#52BCA3",
            "#99C945",
            "#CC61B0",
            "#24796C",
            "#DAA51B",
            "#2F8AC4",
            "#764E9F",
            "#ED645A",
            "#CC3A8E",
            "#A5AA99",
        ]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(title.replace("_", " "))
    list_techniques = list(list_dict_dfs[0].keys())

    color_counter = 0
    for technique in list_techniques:
        val_list = sorted(
            list(dict.fromkeys(list_dict_dfs[0][technique][by].values))
        )
        y_low, y_mid, y_hi = [], [], []

        df_lo, df_mid, df_hi = (
            list_dict_dfs[0][technique],
            list_dict_dfs[1][technique],
            list_dict_dfs[2][technique],
        )

        for val_idx in range(len(val_list)):
            y_low.append(
                df_lo["d"][
                    (df_lo[by] == val_list[val_idx])
                    & (df_lo["randomized"] == "Yes")
                ].values.mean()
            )
            y_mid.append(
                df_mid["d"][
                    (df_mid[by] == val_list[val_idx])
                    & (df_mid["randomized"] == "Yes")
                ].values.mean()
            )
            y_hi.append(
                df_hi["d"][
                    (df_hi[by] == val_list[val_idx])
                    & (df_hi["randomized"] == "Yes")
                ].values.mean()
            )

        s, alpha = 25, 0.10
        plt.scatter(
            range(len(val_list)),
            y_mid,
            c=list_colors[color_counter],
            label=technique,
            s=s,
        )
        plt.plot(
            range(len(val_list)), y_mid, c=list_colors[color_counter],
        )
        plt.fill_between(
            range(len(val_list)),
            y_low,
            y_hi,
            color=list_colors[color_counter],
            alpha=alpha,
        )

        color_counter += 1

    # Set params for plot:
    if by == "knn":
        ticks = [
            r"$\sqrt{N}/20$",
            r"$\sqrt{N}/10$",
            r"$\sqrt{N}/5$",
            r"$\sqrt{N}/2$",
            r"$\sqrt{N}$",
            r"$1.5\sqrt{N}$",
            r"$2\sqrt{N}$",
            r"$4\sqrt{N}$",
        ]
        xlabel = "Number of kNN"
    elif by == "pca":
        ticks = val_list
        xlabel = "Number of PCA components"
    elif by == "w":
        ticks = val_list
        xlabel = "Number of windows for median correction"

    plt.xticks(np.arange(len(val_list)), ticks)
    plt.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    os.makedirs(save_dir + "/png", exist_ok=True)
    os.makedirs(save_dir + "/pdf", exist_ok=True)

    print(
        "Saving to ",
        save_dir
        + "/pdf/{}_library-comparison_{}.pdf".format(
            title.replace(",", ""), org
        ),
    )
    plt.savefig(
        save_dir
        + "/pdf/{}_library-comparison_{}.pdf".format(
            title.replace(",", ""), org
        ),
        format="pdf",
    )
    plt.savefig(
        save_dir
        + "/png/{}_library-comparison_{}.png".format(
            title.replace(",", ""), org
        ),
        format="png",
        dpi=350,
    )


def get_all_pics_dataset(lib_prep, org, dataset, save_dir, plot_random=True):
    df_0_250 = random_noise_parameter(
        lib_prep,
        org,
        dataset,
        save_dir,
        0,
        250,
        what="relative noise",
        by="knn",
    )
    df_250_1000 = random_noise_parameter(
        lib_prep,
        org,
        dataset,
        save_dir,
        250,
        1000,
        what="relative noise",
        by="knn",
    )
    df_1000_5000 = random_noise_parameter(
        lib_prep,
        org,
        dataset,
        save_dir,
        1000,
        5000,
        what="relative noise",
        by="knn",
    )

    plot_scatter_parameter(
        [df_1000_5000, df_250_1000, df_0_250],
        ["1000 - 5000", "250 - 1000", "0 - 250"],
        lib_prep,
        org,
        dataset,
        by="knn",
        title="Noise_distance_based_on_seed,_kNN",
        ylabel="$\\frac{|d_A-d_B|}{|d_A| + |d_B|}$",
        plot_random=plot_random,
    )

    df_0_250 = random_noise_parameter(
        lib_prep,
        org,
        dataset,
        save_dir,
        0,
        250,
        what="relative noise",
        by="pca",
    )
    df_250_1000 = random_noise_parameter(
        lib_prep,
        org,
        dataset,
        save_dir,
        250,
        1000,
        what="relative noise",
        by="pca",
    )
    df_1000_5000 = random_noise_parameter(
        lib_prep,
        org,
        dataset,
        save_dir,
        1000,
        5000,
        what="relative noise",
        by="pca",
    )

    plot_scatter_parameter(
        [df_1000_5000, df_250_1000, df_0_250],
        ["1000 - 5000", "250 - 1000", "0 - 250"],
        lib_prep,
        org,
        dataset,
        by="pca",
        title="Noise_distance_based_on_seed,_PCA",
        ylabel="$\\frac{|d_A-d_B|}{|d_A| + |d_B|}$",
        plot_random=plot_random,
    )

    df_0_250 = random_noise_parameter(
        lib_prep, org, dataset, save_dir, 0, 250, what="overlap", by="knn"
    )
    df_0_1000 = random_noise_parameter(
        lib_prep, org, dataset, save_dir, 0, 1000, what="overlap", by="knn"
    )
    df_0_5000 = random_noise_parameter(
        lib_prep, org, dataset, save_dir, 0, 5000, what="overlap", by="knn"
    )

    plot_scatter_parameter(
        [df_0_5000, df_0_1000, df_0_250],
        ["0 - 5000", "0 - 1000", "0 - 250"],
        lib_prep,
        org,
        dataset,
        step=1,
        by="knn",
        palette="sunsetmid3",
        title="Overlap_of_features_based_on_seed,_kNN",
        ylabel="Overlap",
        plot_random=plot_random,
    )

    df_0_250 = random_noise_parameter(
        lib_prep, org, dataset, save_dir, 0, 250, what="overlap", by="pca"
    )
    df_0_1000 = random_noise_parameter(
        lib_prep, org, dataset, save_dir, 0, 1000, what="overlap", by="pca"
    )
    df_0_5000 = random_noise_parameter(
        lib_prep, org, dataset, save_dir, 0, 5000, what="overlap", by="pca"
    )

    plot_scatter_parameter(
        [df_0_5000, df_0_1000, df_0_250],
        ["0 - 5000", "0 - 1000", "0 - 250"],
        lib_prep,
        org,
        dataset,
        step=1,
        palette="sunsetmid3",
        by="pca",
        title="Overlap_of_features_based_on_seed,_PCA",
        ylabel="Overlap",
        plot_random=plot_random,
    )

    df_violin_0_500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 500, what="overlap", by="knn"
    )
    df_violin_0_1000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 1000, what="overlap", by="knn"
    )
    df_violin_0_2500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 2500, what="overlap", by="knn"
    )
    df_violin_0_5000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 5000, what="overlap", by="knn"
    )

    plot_scatter_parameter(
        [
            df_violin_0_5000,
            df_violin_0_2500,
            df_violin_0_1000,
            df_violin_0_500,
        ],
        ["0 - 5000", "0 - 2500", "0 - 1000", "0 - 500"],
        lib_prep,
        org,
        dataset,
        step=1,
        palette="sunsetmid4",
        by="knn",
        title="kNN_robustness,_overlap",
        ylabel="Overlap",
        plot_random=plot_random,
    )

    df_violin_0_500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 500, what="correlation", by="knn"
    )
    df_violin_0_1000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 1000, what="correlation", by="knn"
    )
    df_violin_0_2500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 2500, what="correlation", by="knn"
    )
    df_violin_0_5000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 5000, what="correlation", by="knn"
    )

    plot_scatter_parameter(
        [
            df_violin_0_5000,
            df_violin_0_2500,
            df_violin_0_1000,
            df_violin_0_500,
        ],
        ["0 - 5000", "0 - 2500", "0 - 1000", "0 - 500"],
        lib_prep,
        org,
        dataset,
        step=1,
        palette="sunsetmid4",
        by="knn",
        title="kNN_robustness,_correlation",
        ylabel="Pearson correlation",
        plot_random=plot_random,
    )

    df_violin_0_500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 500, what="overlap", by="knn"
    )
    df_violin_0_1000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 1000, what="overlap", by="knn"
    )
    df_violin_0_2500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 2500, what="overlap", by="knn"
    )
    df_violin_0_5000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 5000, what="overlap", by="knn"
    )

    plot_scatter_parameter(
        [
            df_violin_0_5000,
            df_violin_0_2500,
            df_violin_0_1000,
            df_violin_0_500,
        ],
        ["0 - 5000", "0 - 2500", "0 - 1000", "0 - 500"],
        lib_prep,
        org,
        dataset,
        step=1,
        palette="sunsetmid4",
        by="knn",
        title="kNN_robustness,_overlap",
        ylabel="Overlap",
        plot_random=plot_random,
    )

    df_violin_0_500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 500, what="overlap", by="pca"
    )
    df_violin_0_1000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 1000, what="overlap", by="pca"
    )
    df_violin_0_2500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 2500, what="overlap", by="pca"
    )
    df_violin_0_5000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 5000, what="overlap", by="pca"
    )

    plot_scatter_parameter(
        [
            df_violin_0_5000,
            df_violin_0_2500,
            df_violin_0_1000,
            df_violin_0_500,
        ],
        ["0 - 5000", "0 - 2500", "0 - 1000", "0 - 500"],
        lib_prep,
        org,
        dataset,
        step=1,
        palette="sunsetmid4",
        by="pca",
        title="PCA_robustness,_overlap",
        ylabel="Overlap",
        plot_random=plot_random,
    )

    df_violin_0_500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 500, what="correlation", by="pca"
    )
    df_violin_0_1000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 1000, what="correlation", by="pca"
    )
    df_violin_0_2500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 2500, what="correlation", by="pca"
    )
    df_violin_0_5000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 5000, what="correlation", by="pca"
    )

    plot_scatter_parameter(
        [
            df_violin_0_5000,
            df_violin_0_2500,
            df_violin_0_1000,
            df_violin_0_500,
        ],
        ["0 - 5000", "0 - 2500", "0 - 1000", "0 - 500"],
        lib_prep,
        org,
        dataset,
        step=1,
        palette="sunsetmid4",
        by="pca",
        title="PCA_robustness,_correlation",
        ylabel="Pearson correlation",
        plot_random=plot_random,
    )

    df_violin_0_500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 500, what="overlap", by="w"
    )
    df_violin_0_1000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 1000, what="overlap", by="w"
    )
    df_violin_0_2500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 2500, what="overlap", by="w"
    )
    df_violin_0_5000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 5000, what="overlap", by="w"
    )

    plot_scatter_parameter(
        [
            df_violin_0_5000,
            df_violin_0_2500,
            df_violin_0_1000,
            df_violin_0_500,
        ],
        ["0 - 5000", "0 - 2500", "0 - 1000", "0 - 500"],
        lib_prep,
        org,
        dataset,
        step=1,
        palette="sunsetmid4",
        by="w",
        title="window_robustness,_overlap",
        ylabel="Overlap",
        plot_random=plot_random,
    )

    df_violin_0_500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 500, what="correlation", by="w"
    )
    df_violin_0_1000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 1000, what="correlation", by="w"
    )
    df_violin_0_2500 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 2500, what="correlation", by="w"
    )
    df_violin_0_5000 = compare_parameter(
        lib_prep, org, dataset, save_dir, 0, 5000, what="correlation", by="w"
    )

    plot_scatter_parameter(
        [
            df_violin_0_5000,
            df_violin_0_2500,
            df_violin_0_1000,
            df_violin_0_500,
        ],
        ["0 - 5000", "0 - 2500", "0 - 1000", "0 - 500"],
        lib_prep,
        org,
        dataset,
        step=1,
        palette="sunsetmid4",
        by="w",
        title="window_robustness,_correlation",
        ylabel="Pearson correlation",
        plot_random=plot_random,
    )
