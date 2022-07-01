#!/bin/bash
user=ubuntu
export PATH=/home/$user/anaconda3/bin:/home/$user/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

source ~/anaconda3/etc/profile.d/conda.sh

cd /home/ubuntu/stockPrediction/StockPrediction

TODAY=$(date "+%Y%m%d")

echo "Executing DeepStockPrediction Web App ${TODAY}"
conda activate stockPrediction
export PYTHONPATH=${PWD}
export FLASK_APP=src/stock_prediction/app/app.py
flask run --host=0.0.0.0 --port=8080

