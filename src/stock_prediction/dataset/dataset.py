from typing import Tuple, List

import numpy as np
import pickle
import torch.utils.data
import torchvision.transforms

from src.stock_prediction.components.utils import read_img


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 x_data: List,
                 y_data: List,
                 img_size: Tuple[int, int]):
        super().__init__()

        self.X_data = x_data
        self.Y_data = y_data
        self.img_size = img_size
        self.len = len(y_data)
        # self.meanRGB, self.stdRGB = self._calculate_mean_std()
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Resize(self.img_size)
                # torchvision.transforms.Normalize(mean=self.meanRGB, std=self.stdRGB)
            ]
        )

    def __getitem__(self, idx: int) -> Tuple:
        with open(self.X_data[idx], "rb") as f:
            X = pickle.load(f)
        return self.transform(X), self.Y_data[idx]

    def __len__(self) -> int:
        return self.len

    def _calculate_mean_std(self):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
        meanRGB = [np.mean(transform(x).numpy(), axis=(1, 2)) for x in self.X_data]
        stdRGB = [np.std(transform(x).numpy(), axis=(1, 2)) for x in self.X_data]

        meanR = np.mean([m[0] for m in meanRGB])
        meanG = np.mean([m[1] for m in meanRGB])
        meanB = np.mean([m[2] for m in meanRGB])

        stdR = np.mean([s[0] for s in stdRGB])
        stdG = np.mean([s[1] for s in stdRGB])
        stdB = np.mean([s[2] for s in stdRGB])

        return (meanR, meanG, meanB), (stdR, stdG, stdB)
