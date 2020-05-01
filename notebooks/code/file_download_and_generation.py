import warnings
warnings.simplefilter(action='ignore')

import scanpy as sc
import numpy as np
import pandas as pd
from scipy.io import mmread
import os


def process_ding(root_dir):
    """
    root_dir should be Ding_2020 with the following structure:
    Ding_2020\
            human\
                cells.read.new.txt
                counts.read.txt.gz
                genes.read.txt
                meta.txt
            mouse\
                cell.names.new.txt
                count.reads.txt.gz
                genes.count.txt
                meta_combined.txt

    The output files will be adatas, one adata for one technique and organism
    """

    # First we process mouse data (cortex)
    matrix = mmread(root_dir + '/mouse/count.reads.txt.gz')
    features = np.loadtxt(root_dir + 'mouse/genes.count.txt', dtype=str)
    barcodes = np.loadtxt(root_dir + 'mouse/cell.names.new.txt', dtype=str)

    adata = sc.AnnData(X=matrix.tocsr()).transpose()
    adata.var_names = features
    adata.obs_names = barcodes

    meta = pd.read_csv(root_dir + '/mouse/meta_combined.txt', sep='\t', skiprows=[1])  # The 2nd row are datatypes
    adata = adata[meta['NAME'].values]
    adata.obs['method'] = meta['Method'].values
    adata.obs['CellType'] = meta['CellType'].values

    methods = list(dict.fromkeys(meta['Method']))
    for method in methods:
        print(f'mouse {method}')
        adata_method = adata[adata.obs['method'] == method]
        print(f'{len(adata_method)} cells selected')
        adata_method.write_h5ad(root_dir + f'/{method}_mouse.h5ad')

    # Now we repeat with human
    matrix = mmread(root_dir + '/human/counts.read.txt.gz')
    features = np.loadtxt(root_dir + 'human/genes.read.txt', dtype=str)
    barcodes = np.loadtxt(root_dir + 'human/cells.read.new.txt', dtype=str)

    adata = sc.AnnData(X=matrix.tocsr()).transpose()
    adata.var_names = features
    adata.obs_names = barcodes

    meta = pd.read_csv(root_dir + '/human/meta.txt', sep='\t', skiprows=[1])  # The 2nd row are datatypes
    adata = adata[meta['NAME'].values]
    adata.obs['method'] = meta['Method'].values
    adata.obs['cell_types'] = meta['CellType'].values

    methods = list(dict.fromkeys(meta['Method']))
    for method in methods:
        print(f'human {method}')
        adata_method = adata[adata.obs['method'] == method]
        print(f'{len(adata_method)} cells selected')
        adata_method.write_h5ad(root_dir + f'/{method}_human.h5ad')


def process_mereu(root_dir):
    """
    In this case, because names are informative, we only need to download the data, read the csv files and output
    the adatas.   
    """
    tsv_dir = root_dir + '/tsv/'
    
    df_cell_types_human = pd.read_csv(root_dir + '/cell_types/human.csv')
#     df_cell_types_mouse = pd.read_csv(root_dir + '/cell_types/mouse.csv')
    
    list_techniques = ['CELseq2', 'Dropseq', 'QUARTZseq', 'SMARTseq2', 'SingleNuclei', 'ddSEQ', 'inDrop', '10X']
    file_list = os.listdir(tsv_dir)
    
    for technique in list_techniques:
        for org in ['human']:  # TODO: add mouse when I have the df
            print(technique, org)
            
            file_select = [f for f in file_list if (technique in f) & (org in f)][0]
            
            adata = sc.read_text(tsv_dir + file_select).transpose()
            adata.var_names_make_unique()
            
            if org == 'human':
                cells_select = np.intersect1d(df_cell_types_human['colnames'].values, adata.obs_names.values)
                cell_types = df_cell_types_human['cell_types'][df_cell_types_human['colnames'].isin(cells_select)].values
            else:
                cells_select = np.intersect1d(df_cell_types_mouse['colnames'].values, adata.obs_names.values)
                cell_types = df_cell_types_mouse['cell_types'][df_cell_types_mouse['colnames'].isin(cells_select)].values
                                                 
            len_before, len_after = len(adata.obs_names), len(cells_select)
            print(f'{len_before} before removal, {len_after} after cell removal.')
            adata = adata[cells_select]
            
            adata.obs['cell_types'] = cell_types
            
            adata.write_h5ad(root_dir + f'{technique}_{org}.h5')


