from dataclasses import dataclass

from .sub_config import *


# Main Configs
@dataclass(frozen=True)
class MainConfig:
    # Connection
    data_base_connection: DataBaseConfig

    # Model
    data_loader: DataLoaderConfig
    preprocessor: PreprocessorConfig
    trainer: TrainerConfig
    evaluator: EvaluatorConfig
    # predictor: PredictorConfig

    # Path
    data_path: str
    out_path: str