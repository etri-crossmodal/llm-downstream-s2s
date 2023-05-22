
"""
    Pytorch-lightning datamodule for KLUE-NLI dataset
"""
import os
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
        super().__init__()
        self.valid_proportion = valid_proportion
        self.batch_size = batch_size
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        # grab KLUE/NLI dataset from prepared jsonl
        basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "klue_datasets/")
        klue_nli_whole = load_dataset(
            path=os.path.join(basepath, "klue_data.py"),
            name="nli", data_dir=basepath)

        # split train into train/valid
        splitted_ds = klue_nli_whole["train"].train_test_split(test_size=self.valid_proportion,)
        self.dataset_train_iter = splitted_ds["train"]
        self.dataset_valid_iter = splitted_ds["test"]

        # use validation dataset split as a test
        self.dataset_test_iter = klue_nli_whole["test"]

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
        super().__init__()
        self.valid_proportion = valid_proportion
        self.batch_size = batch_size
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        # grab KLUE/NLI dataset from prepared jsonl
        basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "klue_datasets/")
        klue_ynat_whole = load_dataset(
            path=os.path.join(basepath, "klue_data.py"),
            name="nli", data_dir=basepath)

        # split train into train/valid
        splitted_ds = klue_ynat_whole["train"].train_test_split(test_size=self.valid_proportion,)
        self.dataset_train_iter = splitted_ds["train"]
        self.dataset_valid_iter = splitted_ds["test"]

        # use validation dataset split as a test
        self.dataset_test_iter = klue_ynat_whole["test"]

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)


class KLUEMRCDataModule(pl.LightningDataModule):
    def __init__(self, valid_proportion: float=0.05,
                 batch_size: int=32,
                 **kwargs):
        super().__init__()
        self.valid_proportion = valid_proportion
        self.batch_size = batch_size
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        # grab KLUE/NLI dataset from prepared jsonl
        basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "klue_datasets/")
        klue_mrc_whole = load_dataset(
            path=os.path.join(basepath, "klue_data.py"),
            name="mrc", data_dir=basepath)

        #klue_mrc_whole['train'] = klue_mrc_whole['train'].remove_columns(["plausible_answer",])
        #klue_mrc_whole['test'] = klue_mrc_whole['test'].remove_columns(["plausible_answer",])

        # split train into train/valid
        splitted_ds = klue_mrc_whole["train"].train_test_split(test_size=self.valid_proportion,)
        self.dataset_train_iter = splitted_ds["train"]
        self.dataset_valid_iter = splitted_ds["test"]

        # use validation dataset split as a test
        self.dataset_test_iter = klue_mrc_whole["test"]

        def filter_longer_answer(example):
            idx = example['answers']['text'].index(min(example['answers']['text'], key=len))
            example['answers'] = {'text': example['answers']['text'][idx],
                                  'start_idx': example['answers']['start_idx'][idx]}
            return example

        # 정답이 여러개 인 것이 있을 수 있으므로, 하나로 줄여낸다.
        # torch.utils.data.DataLoader는 엘리먼트 수가 정렬되어 있어야 함.
        self.dataset_train_iter = self.dataset_train_iter.map(filter_longer_answer)
        self.dataset_valid_iter = self.dataset_valid_iter.map(filter_longer_answer)
        self.dataset_test_iter = self.dataset_test_iter.map(filter_longer_answer)

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)
