from pathlib import Path
from datetime import datetime, date

# Test Requirements Installation
import bs4
import flask
import matplotlib
import mplfinance
import numpy
import pandas
import pandas_datareader
import PIL
import plotly
import sklearn
import torch
import torchvision
import tqdm
import typer
import yfinance

from pytz import timezone

from src.stock_prediction.config.parser import ConfigParser
from src.stock_prediction.components.utils import generate_dir
from src.stock_prediction.service.preprocessor import Preprocessor
from src.stock_prediction.service.trainer import Trainer

pwd = Path(".").absolute()
app = typer.Typer()


@app.command()
def test(
        config_file: str = typer.Option(
            default=f"{pwd}/config/default_config.json",
            help=f"Config file path option. Using default is strongly recommended./n"
        )
):
    config_parser = ConfigParser(config_file)
    config = config_parser.main_config
    generate_dir(config)
    today = date.strftime(datetime.now(timezone('Asia/Seoul')), "%Y-%m-%d")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(config)
    preprocessor = Preprocessor(config, True, True)

    print(f"Current Date: {today}")
    print(f"Available Device for Model Training: {device}")
    print("Initial Build done successfully!")


if __name__ == "__main__":
    app()
