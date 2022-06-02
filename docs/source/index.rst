.. triku documentation master file, created by
   sphinx-quickstart on Thu Jun 25 16:02:29 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Triku - a feature selection method based on nearest neighbors for single-cell data
==================================================
.. image:: https://badge.fury.io/py/triku.svg
    :target: https://badge.fury.io/py/triku

.. image:: https://github.com/alexmascension/triku/actions/workflows/triku_cicd.yml/badge.svg
  :target: https://github.com/alexmascension/triku

.. image:: https://readthedocs.org/projects/triku/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://triku.readthedocs.io/en/latest/?badge=latest

.. image:: https://codecov.io/gh/alexmascension/triku/branch/master/graph/badge.svg?token=XV50UHB80N
   :target: https://codecov.io/gh/alexmascension/triku

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4016715.svg
   :target: https://doi.org/10.5281/zenodo.4016715

.. image:: https://pepy.tech/badge/triku
   :target: https://pepy.tech/project/triku

|
Triku (hedgehog in euskera) is a feature selection method prepared for Single Cell Analysis.
Triku has been prepared to work with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_
``annData`` objects directly.

You can visit the notebooks at out `notebook repo <https://github.com/alexmascension/triku_notebooks>`_

How do I use triku?
^^^^^^^^^^^^^^^^^^^
We suppose you are going to run triku using scanpy. To use triku, simply
import it and run it in one line::

   import triku as tk

   tk.tl.triku(adata)

After that, you can find which features are selected ``adata.var['highly_variable']``.
The scores for each gene are located in ``adata.var['triku_distance']``.
The higher the score, the better.

If you are using scanpy, you **must** run triku before running ``sc.pp.pca`` and
``sc.pp.neighbors``. We recommend running these commands with the following settings::

   import scanpy as sc

   sc.pp.pca(adata)
   sc.pp.neighbors(adata, metric='cosine', n_neighbors=int(0.5 * len(adata) ** 0.5))

You can run triku with raw or log-transformed count matrices. Scores tend to be better
in log-transformed matrices, although the results depend on the dataset.


Cite us!
~~~~~~~~~
If you want to learn more about how triku works, you can read `our paper <https://doi.org/10.1093/gigascience/giac017>`_ [1]_. Don't forget to cite it if you find triku useful!

.. [1] Alex M Ascensión, Olga Ibáñez-Solé, Iñaki Inza, Ander Izeta, Marcos J Araúzo-Bravo, Triku: a feature selection method based on nearest neighbors for single-cell data, GigaScience, Volume 11, 2022, giac017, https://doi.org/10.1093/gigascience/giac017


.. toctree::
   :maxdepth: 1
   :hidden:

   triku-work
   install
   usage
   changelog
