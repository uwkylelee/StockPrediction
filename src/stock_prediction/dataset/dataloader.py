import torch


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self):
        super().__init__()