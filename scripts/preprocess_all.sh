#!/bin/bash

export PYTHONPATH=${PWD}

# generates chart images with mav_line and with volume
python3 src/stock_prediction/main.py preprocess --mav-line --volume
# generates chart images without mav_line and with volume
python3 src/stock_prediction/main.py preprocess --no-mav-line --volume
# generates chart images with mav_line and without volume
python3 src/stock_prediction/main.py preprocess --mav-line --no-volume
# generates chart images without mav_line and without volume
python3 src/stock_prediction/main.py preprocess --no-mav-line --no-volume
