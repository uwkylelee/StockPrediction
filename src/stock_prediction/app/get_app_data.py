import json
import os

from datetime import datetime, date, timedelta
from pytz import timezone

import pandas_datareader as pdr

import torch
import torchvision
import torchvision.models as models

from tqdm import tqdm

from src.stock_prediction.app.plot_candlechart import plot_candlechart
from src.stock_prediction.app.predict import predict
from src.stock_prediction.components.stock_crawler import get_stock_dict

if not os.path.isdir("app/temp"):
    os.mkdir("app/temp")

if not os.path.isdir("app/stock_info"):
    os.mkdir("app/stock_info")

START_DATE = date.strftime(datetime.now(timezone('Asia/Seoul')) - timedelta(days=35), "%Y-%m-%d")
END_DATE = date.strftime(datetime.now(timezone('Asia/Seoul')) - timedelta(days=1), "%Y-%m-%d")

DEVICE = "cuda:2"
MODEL = models.vgg16_bn()
MODEL.to(DEVICE)
model_path = "output/model/vgg16_bn_224_lr9e-05_wd8e-03_p4e-02.pth"
MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))

TRANSFORM = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

STOCK_DICT = get_stock_dict(300)

for market, stocks in STOCK_DICT.items():
    for stock_name, stock_code in tqdm(stocks.items(), desc=f"{market}"):
        stock_df = pdr.DataReader(stock_code, "naver", START_DATE, END_DATE).astype(int)
        if len(stock_df) < 21:
            continue
        close_prev = stock_df.iloc[-2]["Close"]
        close_next = stock_df.iloc[-1]["Close"]
        price_change = close_next / close_prev - 1
        if price_change >= 0.04:
            true_label = 1
        else:
            true_label = 0

        candle_chart = plot_candlechart(stock_df.iloc[-20:].reset_index(), stock_name, stock_code)
        pred, prob = predict(stock_df.iloc[-21:-1], MODEL, TRANSFORM, DEVICE)
        price_format = "+{:.2f}" if price_change >= 0 else "{:.2f}"
        STOCK_DICT[market][stock_name] = {"code": stock_code,
                                          "price_change": price_format.format(price_change * 100),
                                          "candle_chart": candle_chart,
                                          "true_label": true_label,
                                          "pred": pred,
                                          "prob": prob}

today = date.strftime(datetime.now(timezone('Asia/Seoul')) - timedelta(days=1), "%y%m%d")

with open(f"app/stock_info/{today}.json", "w") as json_file:
    json.dump(STOCK_DICT, json_file, indent=4, ensure_ascii=False)
