import scanpy as sc
import pandas as pd
import scipy.sparse as spr
import numpy as np
import os

from ..logg import triku_logger, TRIKU_LEVEL
import logging


def set_level_logger(level):
    dict_levels = {'debug': logging.DEBUG, 'triku': TRIKU_LEVEL, 'info': logging.INFO, 'warning': logging.WARNING,
                   'error': logging.ERROR, 'critical': logging.CRITICAL}

    triku_logger.setLevel(dict_levels[level])
