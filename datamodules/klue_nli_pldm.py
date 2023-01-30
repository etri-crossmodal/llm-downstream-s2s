
"""
    Pytorch-lightning datamodule for KLUE-NLI dataset
"""

import logging

from typing import Any, List, Union, Optional

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, ByT5Tokenizer)
from datasets import (DatasetDict, Dataset, load_dataset, interleave_datasets, load_from_disk)

class KLUENLIDataModule(pl.LightningDataModule):
    def __init__(self, valid_proportion: float=0.01,
                 batch_size: int=32,
                 **kwargs):
        super(KLUENLIDataModule, self).__init__()
        self.valid_proportion = valid_proportion
        self.batch_size = batch_size
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        # grab KLUE/NLI dataset from huggingface datasets
        klue_nli_whole = load_dataset("klue", "nli")

        # split train into train/valid
        splitted_ds = klue_nli_whole["train"].train_test_split(test_size=self.valid_proportion,)
        self.dataset_train_iter = splitted_ds["train"]
        self.dataset_valid_iter = splitted_ds["test"]

        # use validation dataset split as a test
        self.dataset_test_iter = klue_nli_whole["validation"]

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)


class KLUEYNATDataModule(pl.LightningDataModule):
    def __init__(self, valid_proportion: float=0.01,
                 batch_size: int=32,
                 **kwargs):
        super(KLUEYNATDataModule, self).__init__()
        self.valid_proportion = valid_proportion
        self.batch_size = batch_size
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        # grab KLUE/NLI dataset from huggingface datasets
        klue_ynat_whole = load_dataset("klue", "ynat")

        # split train into train/valid
        splitted_ds = klue_ynat_whole["train"].train_test_split(test_size=self.valid_proportion,)
        self.dataset_train_iter = splitted_ds["train"]
        self.dataset_valid_iter = splitted_ds["test"]

        # use validation dataset split as a test
        self.dataset_test_iter = klue_ynat_whole["validation"]

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)
