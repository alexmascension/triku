import scanpy as sc
import pandas as pd

from ..logg import logger


def get_arr_counts_genes(object_triku):
    # Check type of object and return the matrix as corresponded
    if isinstance(object_triku, sc.AnnData):
        arr_counts = object_triku.X
        arr_genes = object_triku.var_names
    elif isinstance(object_triku, pd.DataFrame):
        arr_counts = object_triku.values
        arr_genes = object_triku.columns.values
    else:
        msg = "Accepted object types are scanpy annDatas or pandas DataFrames (columns are genes)."
        logger.error(msg)
        raise TypeError(msg)

    return arr_counts, arr_genes


def get_dict_triku(dict_triku, object_triku):
    msg_not_dict = "We could not initialize 'dict_triku'. Make sure that (1) if object_triku is an annData object, " \
                   ".var['triku_selected_genes'] and .var['triku_entropy'] exist or (2) set 'dict_triku' with the" \
                   "proper dictionary."

    if dict_triku is None:
        if isinstance(object_triku, sc.AnnData):
            if 'triku_entropy' in object_triku.var and 'triku_selected_genes' in object_triku.var:
                dict_triku = {
                    'triku_entropy': dict(zip(object_triku.var_names, object_triku.var['triku_entropy'].values)),
                    'triku_selected_genes': object_triku.var[object_triku.var['triku_selected_genes'] ==
                                                             True].index.tolist()}
            else:
                logger.error(msg_not_dict)
                TypeError(msg_not_dict)
        else:
            logger.error(msg_not_dict)
            TypeError(msg_not_dict)

    return dict_triku
