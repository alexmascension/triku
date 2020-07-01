.. triku documentation master file, created by
   sphinx-quickstart on Thu Jun 25 16:02:29 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Triku - Feature selection for Single Cell Analysis
==================================================
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://readthedocs.org/projects/triku/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://triku.readthedocs.io/en/latest/?badge=latest

.. image:: https://codecov.io/gl/alexmascension/triku/branch/dev/graph/badge.svg
  :target: https://codecov.io/gl/alexmascension/triku


Triku (hedgehog in euskera) is a feature selection method prepared for Single Cell Analysis.
Triku has been prepared to work with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_
``annData`` objects directly, although it also works with pandas DataFrames and can be run via
CLI.

How do I use triku?
^^^^^^^^^^^^^^^^^^^
We suppose you are going to run triku using scanpy. To use triku, simply
import it and run it in one line::

   import triku as tk

   tk.tl.triku(adata)

After that, you can find which features are selected ``adata.var['highly_variable']``.
The scores for each gene are located in ``adata.var['emd_distance']``.
The higher the score, the better.

If you are using scanpy, you should run triku before running ``sc.pp.pca`` and
``sc.pp.neighbors``.
You can run triku with raw or log-transformed count matrices. Scores tend to be better
in log-transformed matrices, although the results depend on the dataset.

.. toctree::
   :maxdepth: 1
   :hidden:

   triku-work
   usage
