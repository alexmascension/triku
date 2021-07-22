import logging


"""
We will add a new deggubing level, TRIKU, which is halfway between DEBUG and INFO.
Ray uses INFO to produce its output, which is considerable, and we trust it. To avoid it, we will create an intermediate
level, TRIKU. The default level will be INFO, but for debugging purposses we will use TRIKU.
"""
TRIKU_LEVEL = (logging.INFO + logging.DEBUG) // 2
logging.addLevelName(TRIKU_LEVEL, "TRIKU")

triku_logger = logging.getLogger()
logging.basicConfig(
    format="\n%(asctime)s - triku - %(levelname)s - %(message)s"
)
triku_logger.setLevel(logging.INFO)
