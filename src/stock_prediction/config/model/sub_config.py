from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PreprocessConfig:
    start_date: str
    end_date: str
    save_path: str
    window: int
    prediction_day: int
    percentage: float
    volume: bool


@dataclass(frozen=True)
class TrainConfig:
    split_ratio: Dict[str, float]
    image_size: int
    batch: int
    optimizer: str
    lr_scheduler: str
    lr: float
    weight_decay: float
    num_epoch: int

# @dataclass(frozen=True)
# class EvaluateConfig:

# @dataclass(frozen=True)
# class PredictConfig:
