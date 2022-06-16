import os
import random

import numpy as np
import torch
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.stock_prediction.components.utils import get_logger, read_img
from src.stock_prediction.config.model.main_config import MainConfig
from src.stock_prediction.dataset.dataset import Dataset

logger = get_logger("info", __name__)


class Trainer(object):
    def __init__(self, config: MainConfig):
        self.config = config.train
        self.save_path = config.output_path / "model"
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        self.image_file_path = config.data_path / config.preprocess.save_path
        self.image_file_path = self.image_file_path / f"image_volume_{self.config.volume}_mav_{self.config.mav_line}"
        self.image_file_path = self.image_file_path / "stock_chart_image_bin"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.img_size = (self.config.image_size, self.config.image_size)

        # Data
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None

        # Model Definition
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None

        # Fix Random Seed for Reproducibility
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(self.config.random_seed)

    def execute(self):
        self._load_data()
        self._split_data()
        self.train_dataset = Dataset(self.X_train, self.Y_train, self.img_size)
        self.valid_dataset = Dataset(self.X_valid, self.Y_valid, self.img_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                                       shuffle=True, pin_memory=True, drop_last=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=64,
                                       shuffle=False, pin_memory=True, drop_last=True)
        self._define_model()
        self._train()

    def _load_data(self):
        self.X = []
        self.Y = []

        label_count = {0: 0, 1: 0, 2: 0}

        for img_path in tqdm(os.listdir(self.image_file_path)):
            if img_path == ".ipynb_checkpoints":
                continue
            if int(img_path.split("_")[-2][1:]) != self.img_size[0]:
                continue

            img_path = str(self.image_file_path / img_path)
            label = int(img_path.split("_")[-1][0])

            if label == 0 and label_count[0] >= 30000:
                continue
            if label == 0:
                continue
            if label == 1 and label_count[1] >= 30000:
                continue
            if label == 2 and label_count[2] >= 30000:
                continue

            self.X.append(img_path)
            if self.config.is_binary:
                self.Y.append(1 if label == 2 else 0)
            else:
                self.Y.append(label)
            label_count[label] += 1

        print(label_count)
        print(len(self.Y))
        print(label_count[1] / len(self.Y))

    def _split_data(self):
        train_ratio = self.config.split_ratio["train"]
        valid_ratio = self.config.split_ratio["valid"] / (1 - train_ratio)
        test_ratio = self.config.split_ratio["test"] / (1 - train_ratio)
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(self.X,
                                                                                  self.Y,
                                                                                  test_size=(1 - train_ratio),
                                                                                  train_size=train_ratio,
                                                                                  stratify=self.Y)
        self.X_valid, self.X_test, self.Y_valid, self.Y_test = train_test_split(self.X_valid,
                                                                                self.Y_valid,
                                                                                test_size=round(test_ratio, 2),
                                                                                train_size=round(valid_ratio, 2),
                                                                                stratify=self.Y_valid)

    def _define_model(self):
        self.model = getattr(models, self.config.model)(pretrained=self.config.pretrained)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        self.opt_func = torch.optim.Adam if self.config.optimizer == "Adam" else torch.optim.SGD
        self.optimizer = self.opt_func(self.model.parameters(), self.config.lr, weight_decay=self.config.weight_decay)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=self.config.lr_gamma)
        print(self.optimizer)
        print(self.lr_scheduler.state_dict())

    def _train(self):
        self.results = []

        for epoch in range(self.config.num_epoch):
            self.model.train()
            train_losses = []
            for batch in tqdm(self.train_loader, desc=f"Epoch[{epoch}] Train Step: "):
                self.optimizer.zero_grad()
                loss = self._train_batch(batch)
                train_losses.append(loss)
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()

            train_loss = torch.stack(train_losses).mean().item()
            valid_loss, valid_acc = self._evaluate(epoch)
            result = {"train_loss": train_loss,
                      "valid_loss": valid_loss,
                      "valid_acc": valid_acc}

            self.results.append(result)

            last_lr = self.lr_scheduler.get_last_lr()

            logger.info("Epoch[{}]: train_loss: {:.4f}, valid_loss: {:.4f}, valid_acc: {:.4f}, lr: {:.4f}".format(
                epoch, result["train_loss"], result["valid_loss"], result["valid_acc"], last_lr[0]))

            self._save_model(epoch)

    def _train_batch(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        out = self.model(images)  # Generate predictions
        loss = self.criterion(out, labels)  # Calculate loss
        return loss

    @torch.no_grad()
    def _evaluate(self, epoch: int):
        self.model.eval()
        valid_losses = []
        valid_accs = []
        for batch in tqdm(self.valid_loader, desc=f"Epoch[{epoch}] Evaluation Step: "):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            out = self.model(images)  # Generate predictions
            loss = self.criterion(out, labels)  # Calculate loss
            acc = self.accuracy(out, labels)  # Calculate accuracy

            valid_losses.append(loss.detach())
            valid_accs.append(acc)

        valid_loss = torch.stack(valid_losses).mean().item()
        valid_acc = torch.stack(valid_accs).mean().item()

        return valid_loss, valid_acc

    def _save_model(self, epoch):
        if epoch == 0 or self.results[-1]["valid_loss"] < min([result["valid_loss"] for result in self.results[:-1]]):
            model_name = f"{self.config.model}_{self.config.image_size}_lr{abs(int(np.log10(self.config.lr)))}"
            torch.save(self.model.state_dict(), self.save_path / f"{model_name}.pth")
            logger.info("Successfully saved model")

    @classmethod
    def accuracy(cls, outputs, labels):
        _, predictions = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

    @classmethod
    def binary_accuracy(cls, outputs, labels):
        predictions = (outputs >= 0.5)
        labels = (labels == 1)
        return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
