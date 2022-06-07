import os
import sys

from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf


from tqdm import tqdm

from src.stock_prediction.config.model.main_config import MainConfig


class Fetcher(object):
    def __init__(self,
                 config: MainConfig):
        self.config = config.fetch
        self.stock_code_dict = {
            "005930": "삼성",
            "000660": "SK하이닉스",
            "035420": "NAVER",
            "005380": "현대차",
            "035720": "카카오",
            "051910": "LG화학",
            "105560": "KB금융",
            "005490": "POSCO홀딩스",
            "055550": "신한지주",
            "003550": "LG"
        }

    def execute(self):
        yf.pdr_override()

        for stock_code, name in tqdm(self.stock_code_dict.items()):
            stock_df = pdr.get_data_yahoo(f"{stock_code}.KS", "2012-05-01", "2022-05-01")

            # 인덱스에 할당된 날짜 데이터를 컬럼으로 이동
            stock_df.reset_index(inplace=True)

            # 이동평균선 생성을 위한 데이터 생성
            stock_df["ma_5"] = stock_df["Close"].rolling(5, 1).mean()
            stock_df["ma_20"] = stock_df["Close"].rolling(20, 1).mean()
            stock_df["ma_60"] = stock_df["Close"].rolling(60, 1).mean()
            # stock_df["ma_120"] = stock_df["Close"].rolling(120, 1).mean()
            # stock_df["ma_240"] = stock_df["Close"].rolling(240, 1).mean()

            # csv파일로 저장
            stock_df.to_csv(stock_data_path / f"{name}_{stock_code}.csv", index=False, encoding="utf-8")
