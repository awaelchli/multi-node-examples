import os
import random
from typing import Optional

import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class FakeImageNetDataset(Dataset):
    def __len__(self):
        return int(1e6)

    def __getitem__(self, item):
        img = torch.rand(3, 224, 224)
        label = random.randint(0, 999)
        return img, label


class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: Optional[str] = None,
        fake_data: bool = True,
        batch_size: int = 4,
        workers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.fake_data = fake_data
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        if self.fake_data:
            train_dataset = FakeImageNetDataset()
        else:
            train_dir = os.path.join(self.data_path, "train")
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            train_dataset = datasets.ImageFolder(
                train_dir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )
        return train_loader

    def val_dataloader(self):
        if self.fake_data:
            val_dataset = FakeImageNetDataset()
        else:
            val_dir = os.path.join(self.data_path, "val")
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            val_dataset = (
                datasets.ImageFolder(
                    val_dir,
                    transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]
                    ),
                ),
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()
