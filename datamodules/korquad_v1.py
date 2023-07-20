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
            for aline in in_f:
                adoc = json.loads(aline)

                # dewrap label[0] - list 타입은 DataLoader를 통해 tuple로 묶이게 된다.
                train_data.append({'id': adoc['id'], 'context': adoc['text1'], 'question': adoc['text2'],
                                   'label': adoc['label'][0]})

        with open(os.path.join(basepath, "korquad_v1.0-valid.jsonl"), "rt", encoding="utf-8") as in_f:
            for aline in in_f:
                adoc = json.loads(aline)

                # dewrap label - list 타입은 DataLoader를 통해 tuple로 묶이게 된다.
                test_data.append({'id': adoc['id'], 'context': adoc['text1'], 'question': adoc['text2'],
                                  'label': adoc['label'][0]})

        train_ds = Dataset.from_pandas(pd.DataFrame(data=train_data))
        test_ds = Dataset.from_pandas(pd.DataFrame(data=test_data))

        # split train into train/valid
        splitted_ds = train_ds.train_test_split(test_size=self.valid_proportion,)
        self.dataset_train_iter = splitted_ds["train"]
        self.dataset_valid_iter = splitted_ds["test"]

        # assign test split
        self.dataset_test_iter = test_ds
        #self.dataset_test_iter = test_ds.shard(num_shards=100, index=99)

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        # same as test_dataloader()
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)

    def valid_rawdataset(self):
        return self.dataset_test_iter

    # dataset을 바로 노출하는 메서드
    def test_rawdataset(self):
        return self.dataset_test_iter

    def predict_rawdataset(self):
        return self.dataset_test_iter
