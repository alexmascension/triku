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

    tk.tl.triku(pbmc)

    sc.pp.pca(pbmc)
    sc.pp.neighbors(pbmc)

This is a basic preprocessing of a dataset. You can run triku either after or before
``sc.pp.log1p``. It usually works better after log transformation.

Running triku from CLI
----------------------

If you don't want to run triku within python, you can run it using the CLI.

Simply run the console command ``triku`` and the name of the file with the counts. This file can be an
``annData`` object file, or a csv file with **cells as rows** and **genes as columns**.:

    triku NAME_OF_FILE

Run ``triku -h`` to see how to invoke the arguments for CLI.



