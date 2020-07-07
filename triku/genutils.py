import io
import logging
import os


def get_cpu_count():
    # adapted from https://stackoverflow.com/questions/1006289
    workers = os.cpu_count()

    if "sched_getaffinity" in dir(os):
        workers = len(os.sched_getaffinity(0))

    return workers


class TqdmToLogger(io.StringIO):
    # Adapted from https://gitlhub.com/tqdm/tqdm/issues/313
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger, level, buf = None, None, ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)

    # to create the bar:
    # tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    # for x in tqdm(range, file=tqdm_out)
