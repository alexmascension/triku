Usage
=====

Basic usage
------------

The simplest use of triku is with a common pipeline of scanpy. We show the
example with the ``pbmc3k`` dataset from scanpy::

    import scanpy as sc
    import triku as tk

    pbmc = sc.datasets.pbmc3k()

    sc.pp.filter_cells(pbmc, min_genes=50)
    sc.pp.filter_genes(pbmc, min_cells=10)
    sc.pp.log1p(pbmc)

    sc.pp.pca(pbmc)
    sc.pp.neighbors(pbmc)

    tk.tl.triku(pbmc)


This is a basic preprocessing of a dataset. You can run triku either after or before
``sc.pp.log1p``. It usually works better after log transformation.

After running triku, results are stored in ``adata.var`` (``triku_distance``, ``highly_variable``), and 
in ``adata.uns['triku_params'][None]``. Also, ``adata.obsm['X_triku']`` contains the matrix with the selected features.
This is relevant to calculate the new round of neighbors or to do PCA again with that data.


Advanced usage
--------------
When using triku, there are more some parameters that can be changed. All of them can be found at the API Reference.

*  ``n_features``: The number of features to be selected. For instance, ``tk.tl.triku(adata, n_features=500)`` would select the first 500 features.
*  ``use_raw``: Uses counts from ``adata.raw``. This, for instance, can be used to select non log-transformed counts. This can be set as ``tk.tl.triku(adata, use_raw=True)``.
*  ``name``: Saves the results with a custom name. For instance, if the name is ``sample``, then the results would be stored in ``adata.var['triku_distance_sample']``, and in ``adata.uns['triku_params']['sample']``.

