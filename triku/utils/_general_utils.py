import scanpy as sc
import pandas as pd
import scipy.sparse as spr
import numpy as np
import os
from ..logg import logger


def get_arr_counts_genes(object_triku):
    logger.info('Obtaining count matrix and gene list.')
    # Check type of object and return the matrix as corresponded

    if isinstance(object_triku, sc.AnnData):
        arr_counts, arr_genes = object_triku.X, object_triku.var_names.values
    elif isinstance(object_triku, pd.DataFrame):
        arr_counts, arr_genes = object_triku.values, object_triku.columns.values.values

    elif isinstance(object_triku, str):
        if object_triku.endswith('h5') or object_triku.endswith('h5ad'):
            try:
                adata = sc.read_10x_h5(object_triku)
            except:
                adata = sc.read_h5ad(object_triku)
            arr_counts, arr_genes = adata.X, adata.var_names.values
        elif object_triku.endswith('loom'):
            loom = sc.read_loom(object_triku)
            arr_counts, arr_genes = loom.X, loom.var_names.values
        elif object_triku.endswith('mtx'):
            try:
                mtx = sc.read_10x_mtx(object_triku)
            except:
                mtx = sc.read_mtx(object_triku)
            arr_counts, arr_genes = mtx.X, mtx.var_names.values
        elif object_triku.endswith('txt') or object_triku.endswith('csv') or object_triku.endswith('tsv'):
            df = pd.read_csv(object_triku, sep=None)
            arr_counts, arr_genes = df.values, df.columns.values
        else:
            msg = "Accepted file formats are h5 / h5ad / loom / mtx for adata files, and txt / csv / tsv for matrices."
            logger.error(msg)
            raise TypeError(msg)

    else:
        msg = "Accepted object types are scanpy annDatas or pandas DataFrames (columns are genes)."
        logger.error(msg)
        raise TypeError(msg)

    if spr.isspmatrix(arr_counts):
        arr_counts = arr_counts.todense()

    if isinstance(arr_counts, np.matrix):
        arr_counts = np.asarray(arr_counts)

    arr_genes = make_genes_unique(arr_genes)

    return arr_counts, arr_genes


def get_dict_triku(dict_triku, dict_triku_path, object_triku):
    msg_not_dict = "We could not initialize 'dict_triku'. Make sure that (1) if object_triku is an annData object, " \
                   ".var['triku_selected_genes'] and .var['triku_entropy'] exist or (2) set 'dict_triku' with the" \
                   "proper dictionary."

    if dict_triku_path != '':
        if not os.path.exists(dict_triku_path + '_entropy.txt') or not os.path.exists(dict_triku_path +
                                                                                      '_selected_genes.txt'):
            msg_path = "The objects {}, {} don't exist. Check the path.".format(dict_triku_path + '_entropy.txt',
                                                                                dict_triku_path + '_selected_genes.txt')
            logger.error(msg_path)
            raise FileNotFoundError(msg_path)

        selected_genes = pd.read_csv(dict_triku_path + '_selected_genes.txt', sep='\t', header=None)[0].values.tolist()
        df_entropy = pd.read_csv(dict_triku_path + '_entropy.txt', sep='\t', header=None)
        dict_triku = {
            'triku_entropy': dict(zip(df_entropy[0].values, df_entropy[1].values)),
            'triku_selected_genes': selected_genes}

    if dict_triku is None:
        if isinstance(object_triku, sc.AnnData):
            if 'triku_entropy' in object_triku.var and 'triku_selected_genes' in object_triku.var:
                dict_triku = {
                    'triku_entropy': dict(zip(object_triku.var_names,
                                              object_triku.var['triku_entropy'].values)),
                    'triku_selected_genes': object_triku.var[object_triku.var['triku_selected_genes'] ==
                                                             True].index.tolist()}
            else:
                logger.error(msg_not_dict)
                TypeError(msg_not_dict)
        else:
            logger.error(msg_not_dict)
            TypeError(msg_not_dict)

    return dict_triku


def save_triku(dict_triku, save_name, object_triku):
    if len(save_name) > 0 or isinstance(object_triku, str):
        if len(save_name) == 0:
            save_name = 'triku_{N}'.format(N=np.random.randint(10, 1000000000))

        logger.info('Saving results in {}'.format(save_name))
        df_entropy = pd.DataFrame({'gene': list(dict_triku['triku_entropy'].keys()), 'ent': list(dict_triku['triku_entropy'].values())})
        df_selected_genes = pd.DataFrame(dict_triku['triku_selected_genes'])

        df_entropy.to_csv('{}_entropy.txt'.format(save_name), header=None, index=None, sep='\t')
        df_selected_genes.to_csv('{}_selected_genes.txt'.format(save_name), header=None, index=None,
                                 sep='\t')


def make_genes_unique(arr):
    labels, counts = np.unique(arr, return_counts=True)
    non_unique_labels = labels[counts > 1]

    if len(non_unique_labels) > 0:
        logger.info("We found {} duplicated genes: {}.\nWe will rename them to make them unique.".format(
            len(non_unique_labels), non_unique_labels))

        for label in non_unique_labels:
            label_idxs = np.argwhere(arr == label).flatten()
            counter = 1
            for idx in label_idxs[1:]:
                arr[idx] = arr[idx] + '-{}'.format(counter)
                counter += 1

    return arr
