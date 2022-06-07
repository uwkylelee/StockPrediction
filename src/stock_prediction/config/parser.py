import json

from pathlib import Path
from datetime import datetime

from .model.main_config import MainConfig
from .model.sub_config import *


class ConfigParser:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.current_datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")

        self.config_dict = self._read_config()
        self.main_config = self._update_config()

    def _read_config(self):
        with open(Path(self.config_file), 'r') as reader:
            config_dict = json.load(reader)
        return config_dict

    def _update_config(self):
        preprocess_config = PreprocessConfig(
            start_date=self.config_dict["preprocess"]["start_date"],
            end_date=self.config_dict["preprocess"]["end_date"],
            save_path=self.config_dict["preprocess"]["save_path"],
            window=self.config_dict["preprocess"]["window"],
            prediction_day=self.config_dict["preprocess"]["prediction_day"],
            percentage=self.config_dict["preprocess"]["percentage"],
            volume=self.config_dict["preprocess"]["volume"]
        )

        train_config = TrainConfig(
            split_ratio=self.config_dict["train"]["split_ratio"],
            image_size=self.config_dict["train"]["image_size"],
            batch=self.config_dict["train"]["batch"],
            optimizer=self.config_dict["train"]["optimizer"],
            lr_scheduler=self.config_dict["train"]["lr_scheduler"],
            lr=self.config_dict["train"]["lr"],
            weight_decay=self.config_dict["train"]["weight_decay"],
            num_epoch=self.config_dict["train"]["num_epoch"]
        )

        data_path = Path(self.config_dict["data_path"]).absolute()
        output_path = Path(self.config_dict["output_path"]).absolute()

        main_config = MainConfig(
            preprocess=preprocess_config,
            train=train_config,
            data_path=data_path,
            output_path=output_path
        )

        return main_config
