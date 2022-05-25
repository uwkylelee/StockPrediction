from argparse import ArgumentParser

from src.stock_prediction.config.parser import ConfigParser


def parse_config():
    parser = ArgumentParser(description="Stock Price Prediction by CNN with candle chart image")
    parser.add_argument("-c", "--config_file", default="../../config/default_config.json", type=str,
                        help="config file path (default: ../../config/default_config.json)")

    args, _ = parser.parse_known_args()

    config_parser = ConfigParser(args=args)
    config = config_parser.main_config

    return config