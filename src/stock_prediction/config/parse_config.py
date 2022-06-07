from argparse import ArgumentParser
from pathlib import Path

from src.stock_prediction.config.parser import ConfigParser


def parse_config():
    pwd = Path(".").absolute()
    parser = ArgumentParser(description="Stock Price Prediction by CNN with candle chart image")
    parser.add_argument("-c", "--config_file", default=f"{pwd}/config/default_config.json", type=str,
                        help=f"config file path (default: {pwd}/config/default_config.json)")

    args, _ = parser.parse_known_args()
    print(args)

    config_parser = ConfigParser(args=args)
    config = config_parser.main_config

    return config