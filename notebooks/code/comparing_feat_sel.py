import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as spr
from palettes_and_cmaps import prism
from matplotlib.lines import Line2D
import gc

def clustering_binary_search(adatax, min_res, max_res, max_depth, seed, n_target_c, features, apply_log=True):
    depth = 0
    adata = adatax.copy()
        
    if apply_log:
        sc.pp.log1p(adata)
    
    adata.var['highly_variable'] = [i in features for i in adata.var_names]
    
    sc.pp.pca(adata, n_comps=35, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=int(0.5 * len(adata) ** 0.5), random_state=seed, metric='cosine')
    
    while depth < max_depth:
        print(f'Depth: {depth}, min res: {min_res}, max res: {max_res}')
        if depth == 0:
            sc.tl.leiden(adata, resolution=min_res, random_state=seed)
            leiden_sol, res_sol = adata.obs['leiden'], min_res
            if len(list(dict.fromkeys(leiden_sol))) == n_target_c:
                break

            sc.tl.leiden(adata, resolution=max_res, random_state=seed)
            leiden_sol, res_sol = adata.obs['leiden'], max_res
            if len(list(dict.fromkeys(leiden_sol))) == n_target_c:
                break
        
        mid_res = 0.5 * (max_res + min_res)
        sc.tl.leiden(adata, resolution=mid_res, random_state=seed)
        leiden_sol, res_sol = adata.obs['leiden'], mid_res
        n_clust_mid = len(list(dict.fromkeys(leiden_sol)))
        if n_clust_mid == n_target_c:
            break
            
        if n_clust_mid > n_target_c:
            max_res = mid_res
        else:
            min_res = mid_res
            
        depth += 1
    
    del adata; gc.collect()
    return leiden_sol, res_sol


def get_max_diff_gene(adata, gene, group_col, per_expressing_cells=0.25, trim=0.025):
    mean_exp_val, mean_exp_val_temp = [], []
    
    groups = sorted(list(set(adata.obs[group_col].values)))
    dict_argwhere = {g: np.argwhere(adata.obs[group_col].values == g).ravel() for g in groups}
    if spr.issparse(adata.X):
        exp_gene = np.asarray(adata[:,gene].X.todense()).ravel()
    else:
        exp_gene = adata[:,gene].X.ravel()
    
    # We will first exclude genes that do not pass a minimum expression threshold in any cluster
    for g in groups:
        exp_group = np.sort(exp_gene[dict_argwhere[g]].ravel())
        mean_exp_val_temp.append(np.mean(exp_group[: int((1 - per_expressing_cells) * len(exp_group))])) # for genes with small expression it may amplify noise
        mean_exp_val.append(np.mean(exp_group[int(trim * len(exp_group)) : int((1 - trim) * len(exp_group))]))

    if np.all(np.isnan(mean_exp_val_temp)) or np.sum(mean_exp_val_temp) == 0:
        return 0, [0] * len(groups)
    
    mean_exp_val_to_1 = np.array(mean_exp_val)/sum(mean_exp_val)
    
    info = max(np.sort(mean_exp_val_to_1)[3] - np.sort(mean_exp_val_to_1)[0], np.sort(mean_exp_val_to_1)[-1] - np.sort(mean_exp_val_to_1)[-4])
    
    return info, mean_exp_val_to_1


def plot_max_var_x_method(df_feature_ranks, df_max_var_dataset, feature_list=[0, 50, 100, 200, 500, 1000], title=''):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    palette = prism
    
    for method_idx in range(len(df_feature_ranks.columns)):
        df_feature_ranks_method = df_feature_ranks[df_feature_ranks.columns[method_idx]]
        for feat_idx in range(len(feature_list) - 2, -1, -1):
            genes_method_feat = df_feature_ranks_method[(df_feature_ranks_method >= feature_list[feat_idx]) & (df_feature_ranks_method < feature_list[feat_idx + 1])].index
            y_vals = df_max_var_dataset.loc[genes_method_feat, 'maximum_variation'].values
            ax.scatter([method_idx + (len(feature_list) // 2 - feat_idx) * 0.125] * len(y_vals), y_vals, color=palette[feat_idx], s=15, alpha=0.35)
    
    
    legend_elements = [Line2D([0], [0], marker='o', color=palette[idx], label=feature_list[idx+1]) for idx in range(len(feature_list) - 1)]
    ax.legend(handles=legend_elements)
    ax.set_xticks(range(len(df_feature_ranks.columns)))
    ax.set_xticklabels(df_feature_ranks.columns, rotation = 45)
    ax.set_ylabel('Maximum difference')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.suptitle(title)
    plt.show()
    
    
def plot_max_var_x_dataset(dict_df_feature_ranks, dict_df_max_var_dataset, n_features=200, title=''):
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
            genes_method_feat = df_method_rank[(df_method_rank >= 0) & (df_method_rank < n_features)].index
            y_vals = df_max_var_dataset.loc[genes_method_feat, 'maximum_variation'].values
            x = x_idx + (len(method_list) // 2 - met_idx) * 0.085
                        
            ax.scatter(x, np.median(y_vals), color=palette[met_idx], s=13)
            ax.plot([x, x], [np.percentile(y_vals, 25), np.percentile(y_vals, 75)], color=palette[met_idx], linewidth=2)
                       
    legend_elements = [Line2D([0], [0], marker='o', color=palette[idx], label=method_list[::-1][idx]) for idx in range(len(method_list))[::-1]]
    ax.legend(handles=legend_elements)
    ax.set_xticks(range(len(list_datasets)))
    ax.set_xticklabels(list_datasets, rotation = 45)
    ax.set_ylabel('Maximum difference')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.suptitle(title)
    plt.show()
    
    
def plot_ARI_x_method(dict_ARI, title='', figsize=(15,8)):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    palette = prism
    
    list_dfs = list(dict_ARI.keys())
    list_methods = dict_ARI[list_dfs[0]].columns
    
    
    
    for x_idx, method in enumerate(list_methods):
        for df_idx, df in enumerate(list_dfs):
            y_vals = dict_ARI[df][method].values
            median, per25, per75 = np.median(y_vals), np.percentile(y_vals, 25), np.percentile(y_vals, 75)
            w = 0.09
            
            x = x_idx + (len(list_dfs) // 2 - df_idx) * w
                        
            ax.scatter([x]*len(y_vals), y_vals, color=palette[df_idx], s=13)
            ax.plot([x, x], [per25, per75], color=palette[df_idx], linewidth=2)
            ax.plot([x-w/2, x+w/2], [median, median], color=palette[df_idx], linewidth=2)
            
            
    legend_elements = [Line2D([0], [0], marker='o', color=palette[idx], label=list_dfs[idx]) for idx in range(len(list_dfs))]
    ax.legend(handles=legend_elements)
    ax.set_xticks(range(len(list_methods)))
    ax.set_xticklabels(list_methods, rotation = 45)
    ax.set_ylabel('ARI')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.suptitle(title)
    plt.show()
    

def plot_ARI_x_dataset(dict_ARI, title='', figsize=(15,8)):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    palette = prism
    
    list_dfs = list(dict_ARI.keys())
    list_methods = dict_ARI[list_dfs[0]].columns
    
    
    for df_idx, df in enumerate(list_dfs):
        for x_idx, method in enumerate(list_methods):
            y_vals = dict_ARI[df][method].values
            median, per25, per75 = np.median(y_vals), np.percentile(y_vals, 25), np.percentile(y_vals, 75)
            w = 0.09
            
            x = df_idx + (len(list_dfs) // 2 - x_idx) * w
                        
            ax.scatter([x]*len(y_vals), y_vals, color=palette[x_idx], s=13)
            ax.plot([x, x], [per25, per75], color=palette[x_idx], linewidth=2)
            ax.plot([x-w/2, x+w/2], [median, median], color=palette[x_idx], linewidth=2)
            
            
    legend_elements = [Line2D([0], [0], marker='o', color=palette[idx], label=list_methods[idx]) for idx in range(len(list_methods)-1, -1, -1)]
    ax.legend(handles=legend_elements)
    ax.set_xticks(range(len(list_dfs)))
    ax.set_xticklabels(list_dfs, rotation = 45)
    ax.set_ylabel('ARI')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.suptitle(title)
    plt.show()