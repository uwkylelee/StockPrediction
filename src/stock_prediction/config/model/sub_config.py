from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PreprocessConfig:
    start_date: str
    end_date: str
    window: int


@dataclass(frozen=True)
class TrainConfig:
    window: int
    prediction_day: int
    percentage: float
    mav_line: bool
    volume: bool
    is_binary: bool
    equalize: bool
    random_seed: int
    split_ratio: Dict[str, float]
    image_size: int
    batch_size: int
    model: str
    pretrained: bool
    optimizer: str
    lr_scheduler: str
    lr_gamma: float
    lr: float
    weight_decay: float
    num_epoch: int

# @dataclass(frozen=True)
# class EvaluateConfig:

# @dataclass(frozen=True)
# class PredictConfig:
