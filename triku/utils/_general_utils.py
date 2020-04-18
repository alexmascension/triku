import scanpy as sc
import pandas as pd
import scipy.sparse as spr
import numpy as np
import os

from ..logg import triku_logger, TRIKU_LEVEL
import logging

def get_arr_counts_genes(object_triku):
    triku_logger.log(TRIKU_LEVEL,  'Obtaining count matrix and gene list.')
    # Check type of object and return the matrix as corresponded

    if isinstance(object_triku, sc.AnnData):
        arr_counts, arr_genes = object_triku.X, object_triku.var_names.values
    elif isinstance(object_triku, pd.DataFrame):
        arr_counts, arr_genes = object_triku.values, object_triku.columns.values.values
    else:
        msg = "Accepted object types are scanpy annDatas or pandas DataFrames (columns are genes)."
        triku_logger.error(msg)
        raise TypeError(msg)

    if spr.isspmatrix(arr_counts):
        arr_counts = arr_counts.todense()

    if isinstance(arr_counts, np.matrix):
        arr_counts = np.asarray(arr_counts)

    make_genes_unique(arr_genes)

    triku_logger.log(TRIKU_LEVEL,  'Array of counts\n{}, shape:{}'.format(arr_counts, arr_counts.shape))
    triku_logger.log(TRIKU_LEVEL,  'Array of genes\n{}'.format(arr_genes))
    return arr_counts, arr_genes


def make_genes_unique(arr):
    labels, counts = np.unique(arr, return_counts=True)
    non_unique_labels = labels[counts > 1]

    if len(non_unique_labels) > 0:
        msg_err = "There are non-unique variable names. Make them unique by setting adata.var_names_make_unique() and" \
                  "run triku again."
        triku_logger.error(msg_err)

        raise BaseException(msg_err)


def set_level_logger(level):
    dict_levels = {'debug': logging.DEBUG, 'triku': TRIKU_LEVEL, 'info': logging.INFO, 'warning': logging.WARNING,
                   'error': logging.ERROR, 'critical': logging.CRITICAL}

    triku_logger.setLevel(dict_levels[level])
