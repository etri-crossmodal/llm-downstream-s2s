"""
    Pytorch-lightning datamodule for KorQUAD v1.0 dataset
"""

import os
import json
import logging
import pandas as pd

from typing import Any, List, Union, Optional

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, ByT5Tokenizer)
from datasets import (DatasetDict, Dataset, load_dataset, interleave_datasets, load_from_disk)

class KorQuadV1DataModule(pl.LightningDataModule):
    def __init__(self, valid_proportion: float=0.05,
                 batch_size: int=16,
                 **kwargs):
        super().__init__()
        self.valid_proportion = valid_proportion
        self.batch_size = batch_size
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        # load jsonl file
        train_data, test_data = [], []
        basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "korquad/")
        with open(os.path.join(basepath, "korquad_v1.0-train.jsonl"), "rt", encoding="utf-8") as in_f:
            docid = 0
            for aline in in_f:
                adoc = json.loads(aline)
                train_data.append({'id': docid, 'context': adoc['text1'], 'question': adoc['text2'],
                                   'label': adoc['label']})
                docid += 1

        with open(os.path.join(basepath, "korquad_v1.0-valid.jsonl"), "rt", encoding="utf-8") as in_f:
            docid = 0
            for aline in in_f:
                adoc = json.loads(aline)
                test_data.append({'id': docid, 'context': adoc['text1'], 'question': adoc['text2'],
                                  'label': adoc['label']})
                docid += 1

        train_ds = Dataset.from_pandas(pd.DataFrame(data=train_data))
        test_ds = Dataset.from_pandas(pd.DataFrame(data=test_data))

        # split train into train/valid
        splitted_ds = train_ds.train_test_split(test_size=self.valid_proportion,)
        self.dataset_train_iter = splitted_ds["train"]
        self.dataset_valid_iter = splitted_ds["test"]

        # assign test split
        self.dataset_test_iter = test_ds

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)
