import json
import os
import random

from datetime import datetime, date, timedelta
from pytz import timezone

from flask import Flask, render_template, request

app = Flask(__name__)

today = date.strftime(datetime.now(timezone('Asia/Seoul')) - timedelta(days=1), "%y%m%d")
with open(f"app/stock_info/{today}.json", "r") as json_file:
    STOCK_DICT = json.load(json_file)

@app.route("/", methods=['POST', 'GET'])
def home():
    params = {}
    args = request.args

    market = args.get("market")
    stock = args.get("stock")
    is_random = args.get("random")

    if market is not None:
        params["stock_list"] = list(STOCK_DICT[market].keys())

    if is_random:
        market = random.choice(["kospi", "kosdaq"])
        params["stock_list"] = list(STOCK_DICT[market].keys())
        stock = random.choice(params["stock_list"])

    params["market"] = market
    params["stock"] = stock

    if stock is not None and STOCK_DICT[market].get(stock) is not None:
        params["price_change"] = STOCK_DICT[market][stock]["price_change"]
        params["candle_chart"] = STOCK_DICT[market][stock]["candle_chart"]
        params["true_label"] = STOCK_DICT[market][stock]["true_label"]
        params["pred"] = STOCK_DICT[market][stock]["pred"]
        params["prob"] = "{:.2f}".format(STOCK_DICT[market][stock]["prob"] * 100)
        params["pred_result"] = params["true_label"] == params["pred"]

    return render_template('index.html', params=params)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8080")
