from itertools import product
import pandas as pd
import triku as tk
from triku.tl._triku_functions import subtract_median
from tqdm.notebook import tqdm
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def run_batch(adata, windows, n_comps, knns, seeds, save_dir, dataset_prefix):
    # We have to run an array of selections. To do that, one of the parameters above must be
    # a list, and the rest a list of one integer. They can also be all lists, but consider that the
    # calculation will take time. Once triku is run, we will export the distances for each combination
    # as csv. Each csv will contain the median-corrected distances, with and without correction with
    # randomization, for a determined combination of window / n_comp / knn and seed (we will use 3 or 5 seeds
    # for replication purposes).

    for window, n_comp, knn, seed in tqdm(product(*[windows, n_comps, knns, seeds])):
        print(window, n_comp, knn, seed)
        tk.tl.triku(adata, n_windows=window, n_comps=n_comp, knn=knn, random_state=seed, verbose='error')

        distances_with_random = adata.var['emd_distance'].values
        mean_exp = adata.X.sum(0)
        distances_without_random = subtract_median(x=mean_exp, y=adata.var['emd_distance_uncorrected'].values,
                                                   n_windows=window)
        print(adata.var['emd_distance_uncorrected'].values[:5])
        df_res = pd.DataFrame(data={'emd_random_correction': distances_with_random,
                                    'emd_no_correction': distances_without_random},
                              index=adata.var_names.values)

        save_file = '{save_dir}/{pref}-w_{w}-comps_{n_comps}-knn_{knn}-seed_{seed}.csv'.format(
            save_dir=save_dir, pref=dataset_prefix, w=window, n_comps=n_comp, knn=knn, seed=seed)
        df_res.to_csv(save_file)


def return_knn_indices(save_dir, org, lib_prep):
    knn_list = []
    for file in os.listdir(save_dir):
        if org in file and lib_prep in file and 'w_100-' in file and 'comps_30-' in file in file:
            knn_str = file[file.find('knn') + 4:]
            knn_list.append(int(knn_str[: knn_str.find('-')]))
    knn_list = list(dict.fromkeys(knn_list))
    return knn_list


def return_pca_indices(save_dir, org, lib_prep):
    # We need to recover the fixed kNN value. This value is the 5th value on the knn_list; so we will take it.
    knn_pinpoint = return_knn_indices(save_dir, org, lib_prep)[4]

    # Now we get the list of n_comps values
    pca_list = []
    for file in os.listdir(save_dir):
        if org in file and lib_prep in file and 'w_100-' in file and 'knn_%s-' % knn_pinpoint in file:
            pca_str = file[file.find('comps') + 6:]
            pca_list.append(int(pca_str[: pca_str.find('-')]))
    pca_list = sorted(list(dict.fromkeys(pca_list)))
    return pca_list, knn_pinpoint


def random_noise_knn(lib_prep, org, save_dir, min_n_feats, max_n_feats):
    list_dists_non_randomized, list_dists_randomized, list_knn = [], [], []

    knn_list = return_knn_indices(save_dir, org, lib_prep)

    for knn in knn_list:
        list_dfs = []
        for file in os.listdir(save_dir):
            if org in file and lib_prep in file and 'w_100-' in file and 'comps_30-' in file and 'knn_' + str(
                    knn) in file:

                df = pd.read_csv(save_dir + file)
                df = df.set_index('Unnamed: 0')
                list_dfs.append(df)

        # find the genes with biggest distance. We will only choose the last dataframe, but for other
        # stuff we will do a combination of all of them
        select_index_df = (df['emd_no_correction'] + df['emd_random_correction']).sort_values(ascending=False).index[
                          min_n_feats:max_n_feats]

        for i in range(len(list_dfs)):
            for j in range(len(list_dfs)):
                if i > j:
                    df_1, df_2 = list_dfs[i], list_dfs[j]
                    list_dists_non_randomized += list((df_1['emd_no_correction'].loc[select_index_df].values -
                                                       df_2['emd_no_correction'].loc[select_index_df].values) / (
                                                              np.abs(df_1['emd_no_correction'].loc[
                                                                         select_index_df].values) +
                                                              np.abs(df_2['emd_no_correction'].loc[
                                                                         select_index_df].values)))
                    list_dists_randomized += list((df_1['emd_random_correction'].loc[select_index_df].values -
                                                   df_2['emd_random_correction'].loc[select_index_df].values) / (
                                                          np.abs(df_1['emd_random_correction'].loc[
                                                                     select_index_df].values) +
                                                          np.abs(df_2['emd_random_correction'].loc[
                                                                     select_index_df].values)))
                    list_knn += [knn] * (max_n_feats - min_n_feats)

    df_violin = pd.DataFrame({'d': np.abs(list_dists_non_randomized + list_dists_randomized),
                              'knn': list_knn * 2,
                              'randomized': ['No'] * len(list_dists_non_randomized) +
                                            ['Yes'] * len(list_dists_randomized)})
    return df_violin


def plot_scatter_random_noise_knn(list_dfs, categories, lib_prep, org, figsize=(7,4)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle("Random noise due to kNN ({}, {})".format(lib_prep, org))
    knn_list = sorted(list(dict.fromkeys(list_dfs[0]['knn'].values)))

    for i in range(len(knn_list)):
        w, sep, s, alpha, N = 0.25, 0.05, 2, 0.30, 25
        list_colors = ["#fcde9c", "#e34f6f", "#7c1d6f"]
        for idx, sub_df in enumerate(list_dfs):
            sub_df_ran = sub_df['d'][(sub_df['knn'] == knn_list[i]) & (sub_df['randomized'] == 'Yes')].values[::N]
            sub_df_no_ran = sub_df['d'][(sub_df['knn'] == knn_list[i]) & (sub_df['randomized'] == 'No')].values[::N]

            ax.scatter(i + sep + w * np.random.rand(len(sub_df_ran)), sub_df_ran, c=list_colors[idx], s=s, alpha=(1 + idx) * alpha)
            ax.scatter(i - sep - w * np.random.rand(len(sub_df_no_ran)), sub_df_no_ran, c=list_colors[idx], s=s, alpha=(1 + idx) * alpha)

        plt.xticks(np.arange(len(knn_list)), ["$\sqrt{N}/20$", "$\sqrt{N}/10$", "$\sqrt{N}/5$", "$\sqrt{N}/2$",
                                              "$\sqrt{N}$ (%s)"%knn_list[4], "$2\sqrt{N}$","$5\sqrt{N}$",
                                              "$10\sqrt{N}$"])

        legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color=list_colors[j], label=categories[j]) for j in range(3)]
        ax.legend(handles=legend_elements, title='N features')
        ax.set_xlabel('Number of kNN')
        ax.set_ylabel('$\\frac{|d_A - d_B|}{|d_A| + |d_B|}$')


def random_noise_pca(lib_prep, org, save_dir, min_n_feats, max_n_feats):
    list_dists_non_randomized, list_dists_randomized, list_pca = [], [], []

    pca_list, knn_pinpoint = return_pca_indices(save_dir, org, lib_prep)

    for pca in pca_list:
        list_dfs = []
        for file in os.listdir(save_dir):
            if org in file and lib_prep in file and 'w_100-' in file and 'comps_%s-' % pca in file and 'knn_%s-' % (
                    knn_pinpoint) in file:

                df = pd.read_csv(save_dir + file)
                df = df.set_index('Unnamed: 0')
                list_dfs.append(df)

        # find the genes with biggest distance. We will only choose the last dataframe, but for other
        # stuff we will do a combination of all of them

        select_index_df = (df['emd_no_correction'] + df['emd_random_correction']).sort_values(ascending=False).index[
                          min_n_feats:max_n_feats]

        for i in range(len(list_dfs)):
            for j in range(len(list_dfs)):
                if i > j:
                    df_1, df_2 = list_dfs[i], list_dfs[j]
                    list_dists_non_randomized += list((df_1['emd_no_correction'].loc[select_index_df].values -
                                                       df_2['emd_no_correction'].loc[select_index_df].values) / (
                                                              np.abs(df_1['emd_no_correction'].loc[
                                                                         select_index_df].values) +
                                                              np.abs(df_2['emd_no_correction'].loc[
                                                                         select_index_df].values)))
                    list_dists_randomized += list((df_1['emd_random_correction'].loc[select_index_df].values -
                                                   df_2['emd_random_correction'].loc[select_index_df].values) / (
                                                          np.abs(df_1['emd_random_correction'].loc[
                                                                     select_index_df].values) +
                                                          np.abs(df_2['emd_random_correction'].loc[
                                                                     select_index_df].values)))
                    list_pca += [pca] * (max_n_feats - min_n_feats)

    df_violin = pd.DataFrame({'d': np.abs(list_dists_non_randomized + list_dists_randomized),
                              'pca': list_pca * 2,
                              'randomized': ['No'] * len(list_dists_non_randomized) +
                                            ['Yes'] * len(list_dists_randomized)})
    return df_violin


def plot_scatter_random_noise_pca(list_dfs, categories, lib_prep, org, figsize=(7,4)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle("Random noise due to PCA number of components ({}, {})".format(lib_prep, org))
    knn_list = sorted(list(dict.fromkeys(list_dfs[0]['pca'].values)))

    for i in range(len(knn_list)):
        w, sep, s, alpha, N = 0.25, 0.05, 2, 0.30, 25
        list_colors = ["#fcde9c", "#e34f6f", "#7c1d6f"]
        for idx, sub_df in enumerate(list_dfs):
            sub_df_ran = sub_df['d'][(sub_df['pca'] == knn_list[i]) & (sub_df['randomized'] == 'Yes')].values[::N]
            sub_df_no_ran = sub_df['d'][(sub_df['pca'] == knn_list[i]) & (sub_df['randomized'] == 'No')].values[::N]

            ax.scatter(i + sep + w * np.random.rand(len(sub_df_ran)), sub_df_ran, c=list_colors[idx], s=s, alpha=(1 + idx) * alpha)
            ax.scatter(i - sep - w * np.random.rand(len(sub_df_no_ran)), sub_df_no_ran, c=list_colors[idx], s=s, alpha=(1 + idx) * alpha)

        plt.xticks(np.arange(len(knn_list)), knn_list)

        legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color=list_colors[j], label=categories[j]) for j in
                           range(3)]
        ax.legend(handles=legend_elements, title='N features')
        ax.set_xlabel('Number of PCA components')
        ax.set_ylabel('$\\frac{|d_A - d_B|}{|d_A| + |d_B|}$')