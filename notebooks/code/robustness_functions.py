from itertools import product
import pandas as pd
import triku as tk
from triku.tl._triku_functions import subtract_median
from tqdm.notebook import tqdm
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts


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
    knn_list = sorted(list(dict.fromkeys(knn_list)))
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


def return_relative_noise(df_1, df_2, select_index_df):
    relative_noise_non_rand = list((df_1['emd_no_correction'].loc[select_index_df].values -
                                    df_2['emd_no_correction'].loc[select_index_df].values) / (
                                           np.abs(df_1['emd_no_correction'].loc[
                                                      select_index_df].values) +
                                           np.abs(df_2['emd_no_correction'].loc[
                                                      select_index_df].values)))
    relative_noise_rand = list((df_1['emd_random_correction'].loc[select_index_df].values -
                                df_2['emd_random_correction'].loc[select_index_df].values) / (
                                       np.abs(df_1['emd_random_correction'].loc[
                                                  select_index_df].values) +
                                       np.abs(df_2['emd_random_correction'].loc[
                                                  select_index_df].values)))
    return relative_noise_rand, relative_noise_non_rand


def return_percentage_overlap(df_1, df_2, min_n_feats, max_n_feats):
    feats_1_no_cor = df_1.sort_values(by='emd_no_correction', ascending=False).index[min_n_feats:max_n_feats].values
    feats_2_no_cor = df_2.sort_values(by='emd_no_correction', ascending=False).index[min_n_feats:max_n_feats].values
    feats_1_rand = df_1.sort_values(by='emd_random_correction', ascending=False).index[min_n_feats:max_n_feats].values
    feats_2_rand = df_2.sort_values(by='emd_random_correction', ascending=False).index[min_n_feats:max_n_feats].values

    percentage_overlap_non_rand = len(np.intersect1d(feats_1_no_cor, feats_2_no_cor)) / (max_n_feats - min_n_feats)
    percentage_overlap_rand = len(np.intersect1d(feats_1_rand, feats_2_rand)) / (max_n_feats - min_n_feats)

    return [percentage_overlap_rand], [percentage_overlap_non_rand]


def return_correlation(df_1, df_2, min_n_feats, max_n_feats):
    feats_1_no_cor = df_1['emd_no_correction'].sort_values(ascending=False).iloc[min_n_feats:max_n_feats].values
    feats_2_no_cor = df_2['emd_no_correction'].sort_values(ascending=False).iloc[min_n_feats:max_n_feats].values
    feats_1_rand = df_1['emd_random_correction'].sort_values(ascending=False).iloc[min_n_feats:max_n_feats].values
    feats_2_rand = df_2['emd_random_correction'].sort_values(ascending=False).iloc[min_n_feats:max_n_feats].values

    correlation_non_rand = sts.pearsonr(feats_1_no_cor, feats_2_no_cor)
    correlation_rand = sts.pearsonr(feats_1_rand, feats_2_rand)

    return [correlation_non_rand[0]], [correlation_rand[0]]


def random_noise_parameter(lib_prep, org, save_dir, min_n_feats, max_n_feats, what, by):
    list_dists_non_randomized, list_dists_randomized, list_param_value = [], [], []

    knn_list = return_knn_indices(save_dir, org, lib_prep)
    pca_list = return_pca_indices(save_dir, org, lib_prep)

    if by == 'knn':
        parameter_list = knn_list
    elif by == 'pca':
        parameter_list, _ = pca_list

    for val in parameter_list:
        list_dfs = []
        for file in os.listdir(save_dir):
            if by == 'knn':
                static_comp = 'w_100-' in file and 'comps_30-' in file
                dyn_comp = 'knn_' + str(val) in file
            elif by == 'pca':
                static_comp = 'w_100-' in file and 'knn_' + str(knn_list[4]) + '-' in file
                dyn_comp = 'comps_{}-'.format(val) in file

            if org in file and lib_prep in file and static_comp and dyn_comp:
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
                    if what == 'relative noise':
                        what_rand, what_non_rand = return_relative_noise(df_1, df_2, select_index_df)
                    elif what == 'overlap':
                        what_rand, what_non_rand = return_percentage_overlap(df_1, df_2, min_n_feats, max_n_feats)
                    else:
                        what_rand, what_non_rand = None, None

                    list_dists_non_randomized += what_non_rand
                    list_dists_randomized += what_rand

                    list_param_value += [val] * len(what_non_rand)

    df_violin = pd.DataFrame({'d': np.abs(list_dists_non_randomized + list_dists_randomized),
                              by: list_param_value * 2,
                              'randomized': ['No'] * len(list_dists_non_randomized) +
                                            ['Yes'] * len(list_dists_randomized)})
    return df_violin


def compare_parameter(lib_prep, org, save_dir, min_n_feats, max_n_feats, what, by):
    list_dists_non_randomized, list_dists_randomized, list_knn = [], [], []

    knn_list = return_knn_indices(save_dir, org, lib_prep)
    pca_list = return_pca_indices(save_dir, org, lib_prep)

    # We first fill on list of dfs with knn = sqrt(N)
    list_dfs_knn_1 = []
    for file in os.listdir(save_dir):
        if org in file and lib_prep in file and 'w_100-' in file and 'comps_30-' in file and 'knn_' + str(
                knn_list[4]) in file:
            df = pd.read_csv(save_dir + file)
            df = df.set_index('Unnamed: 0')
            list_dfs_knn_1.append(df)

    if by == 'knn':
        parameter_list = knn_list
    elif by == 'pca':
        parameter_list, _ = pca_list

    for val in parameter_list:
        list_dfs_knn_2 = []
        for file in os.listdir(save_dir):

            if by == 'knn':
                static_comp = 'w_100-' in file and 'comps_30-' in file
                dyn_comp = 'knn_' + str(val) in file
            elif by == 'pca':
                static_comp = 'w_100-' in file and 'knn_' + str(knn_list[4]) + '-' in file
                dyn_comp = 'comps_{}-'.format(val) in file

            if org in file and lib_prep in file and static_comp and dyn_comp:
                df = pd.read_csv(save_dir + file)
                df = df.set_index('Unnamed: 0')
                list_dfs_knn_2.append(df)

        for i in range(len(list_dfs_knn_1)):
            for j in range(len(list_dfs_knn_2)):
                df_1, df_2 = list_dfs_knn_1[i], list_dfs_knn_2[j]
                if what == 'overlap':
                    what_rand, what_non_rand = return_percentage_overlap(df_1, df_2, min_n_feats, max_n_feats)
                elif what == 'correlation':
                    what_rand, what_non_rand = return_correlation(df_1, df_2, min_n_feats, max_n_feats)
                else:
                    what_rand, what_non_rand = None, None

                list_dists_non_randomized += what_non_rand
                list_dists_randomized += what_rand

                list_knn += [val] * len(what_non_rand)

    df_violin = pd.DataFrame({'d': np.abs(list_dists_non_randomized + list_dists_randomized),
                              by: list_knn * 2,
                              'randomized': ['No'] * len(list_dists_non_randomized) +
                                            ['Yes'] * len(list_dists_randomized)})
    return df_violin


def plot_scatter_parameter(list_dfs, categories, lib_prep, org, by, figsize=(7, 4), N=25, palette='sunsetcontrast3'):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle("PONER TITULO!")
    # fig.suptitle("Random noise due to kNN ({}, {})".format(lib_prep, org))
    val_list = sorted(list(dict.fromkeys(list_dfs[0][by].values)))
    # Set params for plot:
    if by == 'knn':
        ticks = ["$\sqrt{N}/20$", "$\sqrt{N}/10$", "$\sqrt{N}/5$", "$\sqrt{N}/2$",
                                              "$\sqrt{N}$ (%s)" % val_list[4], "$2\sqrt{N}$", "$5\sqrt{N}$",
                                              "$10\sqrt{N}$"]
        xlabel = 'Number of kNN'
    if by == 'pca':
        ticks = val_list
        xlabel = 'Number of PCA components'

    for val_idx in range(len(val_list)):
        w, sep, s, alpha = 0.25, 0.05, 2, 0.20
        if palette == 'sunsetcontrast3':
            list_colors = ["#fcde9c", "#e34f6f", "#7c1d6f"]
        elif palette == 'sunsetmid3':
            list_colors = ["#faa476", "#dc3977", "#7c1d6f"]
        elif palette == 'sunsetmid4':
            list_colors = ["#faa476", "#f0746e", "#dc3977", "#7c1d6f"]
        for df_idx, sub_df in enumerate(list_dfs):
            sub_df_ran = sub_df['d'][(sub_df[by] == val_list[val_idx]) & (sub_df['randomized'] == 'Yes')].values[::N]
            sub_df_no_ran = sub_df['d'][(sub_df[by] == val_list[val_idx]) & (sub_df['randomized'] == 'No')].values[::N]

            alpha_idx = np.round(alpha + (1 - alpha) * (1 + df_idx) / (len(list_dfs)), 2)  # wild 1.0000000000002 LOL
            ax.scatter(val_idx + sep + w * np.random.rand(len(sub_df_ran)), sub_df_ran, c=list_colors[df_idx], s=s,
                       alpha=alpha_idx)
            ax.scatter(val_idx - sep - w * np.random.rand(len(sub_df_no_ran)), sub_df_no_ran, c=list_colors[df_idx],
                       s=s, alpha=alpha_idx)

    plt.xticks(np.arange(len(val_list)), ticks)
    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color=list_colors[j], label=categories[j]) for
                       j in range(len(list_colors))]
    ax.legend(handles=legend_elements, title='N features')
    ax.set_xlabel(xlabel)

    # ax.set_ylabel('$\\frac{|d_A - d_B|}{|d_A| + |d_B|}$')

