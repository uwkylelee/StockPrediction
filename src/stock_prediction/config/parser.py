import json

from argparse import Namespace
from pathlib import Path
from datetime import datetime

from .model.main_config import MainConfig
from .model.sub_config import *


class ConfigParser:
    def __init__(self, args: Namespace):
        self.config_file: str = args.config_file
        self.current_datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")

        self.config_dict = self._read_config()
        self.main_config = self._update_config()

    def _read_config(self):
        with open(Path(self.config_file), 'r') as reader:
            config_dict = json.load(reader)
        return config_dict

    def _update_config(self):
        data_loader_config = FetcherConfig(
            start_date=self.config_dict["data_loader"]["start_date"],
            end_date=self.config_dict["data_loader"]["end_date"],
            reg_exp=self.config_dict["data_loader"]["reg_exp"],
            normalization=self.config_dict["data_loader"]["normalization"]
        )

        preprocessor_config = PreprocessorConfig(
            split_ratio=self.config_dict["preprocessor"]["split_ratio"],
            tokenizer=self.config_dict["preprocessor"]["tokenizer"],
            bpe_vocab_size=self.config_dict["preprocessor"]["bpe_vocab_size"],
            min_vocab_frequency=self.config_dict["preprocessor"]["min_vocab_frequency"],
            max_vocab_percentage=self.config_dict["preprocessor"]["max_vocab_percentage"]
        )

        trainer_config = TrainerConfig(
            corpus_file_name=self.config_dict["trainer"]["corpus_file_name"],
            dictionary_file_name=self.config_dict["trainer"]["dictionary_file_name"],
        )

        evaluator_config = EvaluatorConfig(
            min_num_topics=self.config_dict["evaluator"]["min_num_topics"],
            max_num_topics=self.config_dict["evaluator"]["max_num_topics"],
            add_n_topics=self.config_dict["evaluator"]["add_n_topics"]
        )

        # predictor_config = PredictorConfig(
        #
        # )

        data_path = Path(self.config_dict["data_path"]).absolute()
        out_path = Path(self.config_dict["out_path"]).absolute()

        main_config = MainConfig(
            data_base_connection=db_config,
            data_loader=data_loader_config,
            preprocessor=preprocessor_config,
            trainer=trainer_config,
            evaluator=evaluator_config,
            # predictor=,
            data_path=str(data_path),
            out_path=str(out_path)
        )

        return main_config