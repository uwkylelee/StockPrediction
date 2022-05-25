import os
import sys

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from tqdm import tqdm


class Preprocessor:
    def __init__(self):
        self.data_path = Path.cwd().parent.parent / "data"

    def preprocess(self):
        stock_data_path = self.data_path / "stock_data/daily_stock_data"

        for data in os.listdir(stock_data_path)[:1]:

            stock_df = pd.read_csv(stock_data_path / data, encoding="utf-8")
            stock_df = stock_df[stock_df.columns[1:]]
            stock_df.rename(columns={"date": "Date",
                                     "open": "Open",
                                     "high": "High",
                                     "low": "Low",
                                     "close": "Close",
                                     "volume": "Volume"},
                            inplace=True)
            stock_df["Date"] = stock_df["Date"].apply(lambda x: datetime.strptime(str(x), "%Y%m%d"))
            stock_df.set_index("Date", inplace=True, drop=True)

            file_prefix = data.split(".")[0]
            image_file_path = self.data_path / "stock_chart_image"

            if not os.path.isdir(image_file_path):
                os.mkdir(image_file_path)

            self.save_chart_image(stock_df, image_file_path, file_prefix)

    def save_chart_image(self,
                         df: pd.DataFrame,
                         save_path: Path,
                         file_prefix: str,
                         window=20):
        for i in tqdm(range(len(df) - window), desc=f"Data {file_prefix}"):
            target_df = df[i: i + window]
            label = self.label_data(df, i + window - 1, 5, 0.05)

            ma_5 = mpf.make_addplot(target_df.ma_5, width=1)
            ma_20 = mpf.make_addplot(target_df.ma_20, width=1)
            ma_60 = mpf.make_addplot(target_df.ma_60, width=1)
            ma_120 = mpf.make_addplot(target_df.ma_120, width=1)
            ma_240 = mpf.make_addplot(target_df.ma_240, width=1)

            mpf.plot(target_df,
                     type="candle",
                     style="yahoo",
                     figsize=(7, 7),
                     volume=True,
                     addplot=[ma_5, ma_20, ma_60, ma_120, ma_240],
                     fontscale=0,
                     savefig=save_path / f"{file_prefix}_{i}_{label}.png")

    @classmethod
    def label_data(cls,
                   df: pd.DataFrame,
                   row_num: int,
                   days: int,
                   percentage: float):
        if df["Close"][row_num] * (1 - percentage) > df["Close"][row_num + days]:
            # Consider as decrease in price
            return 1
        elif df["Close"][row_num] * (1 + percentage) < df["Close"][row_num + days]:
            # Consider as increase in price
            return 2
        else:
            # Consider as no change in price
            return 0
