
"""
    Pytorch-lightning datamodule for KorNLI (arXiv:2004.03289)
"""

import os
import logging

from typing import Any, List, Union, Optional

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, ByT5Tokenizer)
from datasets import (DatasetDict, Dataset, load_dataset, interleave_datasets, load_from_disk)

class KorNLIDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int=32,
                 **kwargs):
        super(KorNLIDataModule, self).__init__()
        self.batch_size = batch_size
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        # grab KLUE/NLI dataset from huggingface datasets
        kor_nli_whole = load_dataset(os.path.dirname(os.path.abspath(__file__)) + "/kor_nlu/kor_nlu.py", "nli")

        self.dataset_train_iter = kor_nli_whole["train"]
        self.dataset_valid_iter = kor_nli_whole["validation"]

        # use validation dataset split as a test
        self.dataset_test_iter = kor_nli_whole["test"]

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)
