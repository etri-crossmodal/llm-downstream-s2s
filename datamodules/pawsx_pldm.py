
"""
    Pytorch-lightning datamodule for PAWS-X (arXiv:1908.11828)
"""

import logging

from typing import Any, List, Union, Optional

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, ByT5Tokenizer)
from datasets import (DatasetDict, Dataset, load_dataset, interleave_datasets, load_from_disk)

class paws_xDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int=32,
                 **kwargs):
        super(paws_xDataModule, self).__init__()
        self.batch_size = batch_size
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        # grab PAWS-X ko dataset from huggingface datasets
        whole = load_dataset("paws-x", "ko")

        self.dataset_train_iter = whole["train"]
        self.dataset_valid_iter = whole["validation"]
        self.dataset_test_iter = whole["test"]

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        # same as test_dataloader()
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)

    # dataset을 바로 노출하는 메서드
    def test_rawdataset(self):
        return self.dataset_test_iter

    def predict_rawdataset(self):
        return self.dataset_test_iter
