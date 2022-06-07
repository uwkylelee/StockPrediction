import logging
import os
import sys

from typing import Tuple
from PIL import Image

import cv2

from src.stock_prediction.config.model.main_config import MainConfig


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


def generate_dir(config: MainConfig):
    if not os.path.exists(config.data_path):
        os.mkdir(config.data_path)

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)


def read_img(img_path: str, size: Tuple[int, int]) -> Image.Image:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = Image.fromarray(img)

    return img
