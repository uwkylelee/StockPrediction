import os

from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import pandas_datareader as pdr
import pickle
import yfinance as yf

from tqdm import tqdm

from src.stock_prediction.components.utils import get_logger, read_img
from src.stock_prediction.config.model.main_config import MainConfig

logger = get_logger("info", __name__)
matplotlib.use("agg")


class Preprocessor:
    def __init__(self,
                 config: MainConfig,
                 mav_line: bool,
                 volume: bool):
        self.logger = logger
        self.config = config.preprocess
        self.mav_line = mav_line
        self.volume = volume
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
        self.save_path = config.data_path / self.config.save_path
        self.stock_data_path: Path = self.save_path / "daily_stock_data"
        self.image_path = self.save_path / f"image_volume_{self.volume}_mav_{self.mav_line}"
        self.image_file_path: Path = self.image_path / "stock_chart_image"
        self.image_bin_file_path: Path = self.image_path / "stock_chart_image_bin"

    def execute(self, fetch=False):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        if not os.path.isdir(self.stock_data_path):
            os.mkdir(self.stock_data_path)

        if not os.path.isdir(self.image_path):
            os.mkdir(self.image_path)

        if not os.path.isdir(self.image_file_path):
            os.mkdir(self.image_file_path)

        if not os.path.isdir(self.image_bin_file_path):
            os.mkdir(self.image_bin_file_path)

        if fetch:
            self._fetch_stock_data()
            self._save_chart_image_fetched_data()
        else:
            self._save_chart_image_known_data()

    def _fetch_stock_data(self):
        yf.pdr_override()
        for stock_code, name in tqdm(self.stock_code_dict.items()):
            stock_df = pdr.get_data_yahoo(f"{stock_code}.KS", "2012-05-01", "2022-05-01")

            # 인덱스에 할당된 날짜 데이터를 컬럼으로 이동
            stock_df.reset_index(inplace=True)

            # 이동평균선 생성을 위한 데이터 생성
            stock_df["ma_5"] = stock_df["Close"].rolling(5, 1).mean()
            stock_df["ma_20"] = stock_df["Close"].rolling(20, 1).mean()
            stock_df["ma_60"] = stock_df["Close"].rolling(60, 1).mean()

            # csv파일로 저장
            stock_df.to_csv(self.stock_data_path / f"{name}_{stock_code}.csv", index=False, encoding="utf-8")

    def _save_chart_image_fetched_data(self):
        self.logger.info("Saving_chart_images")
        for stock_code, name in self.stock_code_dict.items():
            stock_df = pd.read_csv(self.stock_data_path / f"{name}_{stock_code}.csv", encoding="utf-8")
            stock_df["Date"] = stock_df["Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
            stock_df.set_index("Date", inplace=True, drop=True)

            file_prefix = f"{name}_{stock_code}"

            if not os.path.isdir(self.image_file_path):
                os.mkdir(self.image_file_path)

            self._generate_chart_image(stock_df, file_prefix)

    def _save_chart_image_known_data(self):
        self.logger.info("Saving_chart_images")
        for data in os.listdir(self.stock_data_path)[:100]:
            stock_df = pd.read_csv(self.stock_data_path / data, encoding="utf-8")
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

            file_prefix = str(data.split(".")[0])

            if not os.path.isdir(self.image_file_path):
                os.mkdir(self.image_file_path)

            self._generate_chart_image(stock_df, file_prefix)

    def _generate_chart_image(self,
                              df: pd.DataFrame,
                              file_prefix: str) -> None:
        edge_color = mpf.make_marketcolors(up="g", down="r", wick="inherit", volume="inherit", edge="None")
        custom_style = mpf.make_mpf_style(base_mpf_style='yahoo', figcolor="black", marketcolors=edge_color)
        width_config = {"candle_linewidth": 6,
                        "candle_width": 0.97,
                        "volume_width": 0.8}

        for i in tqdm(range(len(df) - self.config.window - self.config.prediction_day), desc=f"Data {file_prefix}"):
            target_df = df[i: i + self.config.window]
            if not self.is_valid_data(target_df):
                continue

            label = self.label_data(df, i + self.config.window - 1, 5, self.config.percentage)

            ma_5 = mpf.make_addplot(target_df.ma_5, width=2)
            ma_20 = mpf.make_addplot(target_df.ma_20, width=2)
            ma_60 = mpf.make_addplot(target_df.ma_60, width=2)

            save_path = self.image_file_path / f"{file_prefix}_{i}_{label}.png"

            fig, _ = mpf.plot(target_df,
                              type="candle",
                              style=custom_style,
                              addplot=[ma_5, ma_20, ma_60] if self.mav_line else [],
                              update_width_config=width_config,
                              figsize=(10, 10),
                              fontscale=0,
                              axisoff=True,
                              volume=self.volume,
                              tight_layout=True,
                              returnfig=True,
                              closefig=True)

            # Save image as .png
            fig.savefig(save_path, bbox_inches="tight")
            del (fig, _)

            # Save PIL image as pickle for faster training step
            if os.path.isfile(self.image_bin_file_path / f"{file_prefix}_{i}_s244_{label}.pkl"):
                os.remove(self.image_bin_file_path / f"{file_prefix}_{i}_s244_{label}.pkl")
            for size in [50, 224]:
                pkl_save_path = self.image_bin_file_path / f"{file_prefix}_{i}_s{str(size)}_{label}.pkl"
                with open(pkl_save_path, "wb") as f:
                    pickle.dump(read_img(save_path, (size, size)), f)

    @classmethod
    def label_data(cls,
                   df: pd.DataFrame,
                   row_num: int,
                   days: int,
                   percentage: float):
        if df["Close"][row_num] * (1 - percentage) >= df["Close"][row_num + days]:
            # Consider as decrease in price
            return 1
        elif df["Close"][row_num] * (1 + percentage) <= df["Close"][row_num + days]:
            # Consider as increase in price
            return 2
        else:
            # Consider as no change in price
            return 0

    @classmethod
    def is_valid_data(cls,
                      df: pd.DataFrame):
        """
        캔들 차트 생성시, 해당 데이터 내에 거래량이 0인 거래일이 3일 이상인 경우, 학습 및 예측에 사용하기 부적합한 데이터로 판단하여,
        차트 생성을 하지 않도록 하는 함수 (적합: True, 부적합: False)

        :param df: 차트 생성 대상 주식 거래 데이터
        :return: bool
        """
        # ohlc = ["Open", "High", "Low", "Close"]
        # if (df[ohlc][-3:-2].to_numpy() == df[ohlc][-2:].to_numpy()).all():
        #     return False

        if df.Volume.apply(lambda x: 1 if x == 0 else None).count() >= 3:
            return False
        else:
            return True
