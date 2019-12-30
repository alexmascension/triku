import scanpy as sc
import pandas as pd
import scipy.sparse as spr

from ..logg import logger


def get_arr_counts_genes(object_triku):
    logger.info('Obtaining count matrix and gene list.')
    # Check type of object and return the matrix as corresponded




    if isinstance(object_triku, sc.AnnData):
        arr_counts, arr_genes = object_triku.X, object_triku.var_names
    elif isinstance(object_triku, pd.DataFrame):
        arr_counts, arr_genes = object_triku.values, object_triku.columns.values

    elif isinstance(object_triku, str):
        if object_triku.endswith('h5') or object_triku.endswith('h5ad'):
            try:
                adata = sc.read_10x_h5(object_triku)
            except:
                adata = sc.read_h5ad(object_triku)
            arr_counts, arr_genes = adata.X, adata.var_names
        elif object_triku.endswith('loom'):
            loom = sc.read_loom(object_triku)
            arr_counts, arr_genes = loom.X, loom.var_names
        elif object_triku.endswith('mtx'):
            try:
                mtx = sc.read_10x_mtx(object_triku)
            except:
                mtx = sc.read_mtx(object_triku)
            arr_counts, arr_genes = mtx.X, mtx.var_names
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


def save_triku(dict_triku, save_dir, save_name, object_triku):
    if len(save_dir) > 0 or len(save_name) > 0 or isinstance(object_triku, str):
        if len(save_name) == 0:
            save_name = 'triku_{N}'.format(N=np.random.randint(10, 1000000000))
        if save_dir[-1] == '/': save_dir = save_dir[:-1]

        logger.info('Saving results in {}/{}'.format(save_dir, save_name))

        df_entropy = pd.DataFrame(dict_triku['triku_entropy'])
        df_selected_genes = pd.DataFrame(dict_triku['triku_selected_genes'])

        df_entropy.to_csv('{}/{}_entropy.txt'.format(save_dir, save_name), header=None, index=None, sep='\t')
        df_selected_genes.to_csv('{}/{}_selected_genes.txt'.format(save_dir, save_name), header=None, index=None,
                                 sep='\t')
