from dataclasses import dataclass
from pathlib import Path

from .sub_config import *


# Main Configs
@dataclass(frozen=True)
class MainConfig:
    # Model
    preprocess: PreprocessConfig
    train: TrainConfig
    # evaluate: EvaluateConfig
    # predict: PredictConfig

    # Path
    data_path: Path
    output_path: Path
