import logging

from ..logg import TRIKU_LEVEL
from ..logg import triku_logger


def set_level_logger(level):
    dict_levels = {
        "debug": logging.DEBUG,
        "triku": TRIKU_LEVEL,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    triku_logger.setLevel(dict_levels[level])
