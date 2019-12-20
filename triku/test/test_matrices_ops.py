import pytest

import logging
import numpy as np
import time

import scanpy as sc
import triku as tk

import scipy.sparse as sps

from triku.logg import logger
logger.setLevel(logging.DEBUG)


@pytest.mark.parallel
def get_mean():
    a = np.array([[0, 0, 1, 1, 1, 2],[0, 1, 1, 1, 2, 1]])
    assert tk.utils.return_mean(a) == np.array([0, 0.5, 1, 1, 1.5, 1.5])

    aa = sps.csr_matrix(a)
    assert tk.utils.return_mean(aa) == np.array([0, 0.5, 1, 1, 1.5, 1.5])

    aaa = sps.csc_matrix(a)
    assert tk.utils.return_mean(aaa) == np.array([0, 0.5, 1, 1, 1.5, 1.5])


@pytest.mark.parallel
def get_proporion_zeros():
    a = np.array([[0, 0, 1, 1, 1, 2], [0, 1, 1, 1, 2, 1]])
    assert tk.utils.return_proportion_zeros(a) == np.array([1, 0.5, 0, 0, 0])

    aa = sps.csr_matrix(a)
    assert tk.utils.return_proportion_zeros(aa) == np.array([1, 0.5, 0, 0, 0])

    aaa = sps.csc_matrix(a)
    assert tk.utils.return_proportion_zeros(aaa) == np.array([0, 0.5, 1, 1, 1.5, 1.5])


