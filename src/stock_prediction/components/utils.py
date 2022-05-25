import logging
import sys


def get_logger(log_level: str, name: str):
    formatter = logging.Formatter(
        fmt="%(asctime)s,%(msecs)03dZ | %(name)-35s | %(levelname)-5s | %(funcName)s() : %(msg)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if log_level.lower() == "debug":
        logger.setLevel(logging.DEBUG)
    elif log_level.lower() == "info":
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    logger.addHandler(handler)

    return logger
