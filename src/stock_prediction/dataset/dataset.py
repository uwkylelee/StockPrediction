from typing import Tuple

import numpy as np
import torch.utils.data
import torchvision.transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 x_data: np.ndarray,
                 y_data: np.ndarray,
                 transform: torchvision.transforms):
        super().__init__()

        self.X_data = x_data
        self.Y_data = y_data
        self.len = len(y_data)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple:
        return self.transform(self.X_data[idx]), self.Y_data[idx]

    def __len__(self) -> int:
        return self.len
