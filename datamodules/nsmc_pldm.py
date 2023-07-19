"""
    Pytorch-lightning datamodule for NSMC datasets
"""

import logging

from typing import Any, List, Union, Optional

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, ByT5Tokenizer)
from datasets import (DatasetDict, Dataset, load_dataset, interleave_datasets, load_from_disk)

class NSMCDataModule(pl.LightningDataModule):
    def __init__(self, valid_proportion: float=0.02,
                 batch_size: int=32,
                 **kwargs):
        super(NSMCDataModule, self).__init__()
        self.valid_proportion = valid_proportion
        self.batch_size = batch_size
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        # grab NSMC dataset from huggingface datasets
        nsmc_whole = load_dataset("nsmc")

        # split train into train/valid
        splitted_ds = nsmc_whole["train"].train_test_split(test_size=self.valid_proportion,)
        self.dataset_train_iter = splitted_ds["train"]
        self.dataset_valid_iter = splitted_ds["test"]

        # assign test split
        self.dataset_test_iter = nsmc_whole["test"]

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)

    def test_rawdataset(self):
        return self.dataset_test_iter

    def predict_rawdataset(self):
        return self.dataset_test_iter
