#!/bin/bash
user=ubuntu
project_path=stockPrediction/StockPrediction
export PATH=/home/$user/anaconda3/bin:/home/$user/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

source ~/anaconda3/etc/profile.d/conda.sh

cd /home/$user/$project_path

if [ ! -d app ]
then
  mkdir app
  mkdir app/log
fi

TODAY=$(date "+%Y%m%d")

echo "Generating Stock Data: Stock Crawling & Model Prediction ${TODAY}"
conda activate stockPrediction
export PYTHONPATH=${PWD}
python3 src/stock_prediction/app/get_app_data.py

