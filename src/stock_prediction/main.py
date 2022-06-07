#!/usr/bin/env python
from pathlib import Path

import typer

from src.stock_prediction.config.parser import ConfigParser
from src.stock_prediction.components.utils import generate_dir, get_logger
from src.stock_prediction.service.preprocessor import Preprocessor
from src.stock_prediction.service.trainer import Trainer

logger = get_logger("info", __name__)

pwd = Path(".").absolute()
app = typer.Typer()


@app.command()
def fetch(
        config_file: str = typer.Option(
            default=f"{pwd}/config/default_config.json",
            help=f"config file path (default: {pwd}/config/default_config.json)"
        )
):
    config_parser = ConfigParser(config_file)
    config = config_parser.main_config
    generate_dir(config)


@app.command()
def preprocess(
        config_file: str = typer.Option(
            default=f"{pwd}/config/default_config.json",
            help=f"Config file path option. Using default is strongly recommended."
        ),
        fetch_data: bool = typer.Option(
            default=False,
            help=f"If True, the preprocessor will fetch Top 10 KOSPI stock data from yahoo finance."
                 f"If False, the local stock data in {pwd}/data/stock_data/daily_stock_data will be used."
        )
):
    config_parser = ConfigParser(config_file)
    config = config_parser.main_config
    generate_dir(config)

    logger.info("Run preprocessor.")
    preprocessor = Preprocessor(config)
    preprocessor.execute(fetch_data)


@app.command()
def train(
        config_file: str = typer.Option(
            default=f"{pwd}/config/default_config.json",
            help=f"Config file path option. Using default is strongly recommended."
        )
):
    config_parser = ConfigParser(config_file)
    config = config_parser.main_config
    generate_dir(config)

    logger.info("Run trainer.")
    trainer = Trainer(config)
    trainer.execute()


if __name__ == "__main__":
    app()
