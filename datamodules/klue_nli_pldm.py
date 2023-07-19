
"""
    Pytorch-lightning datamodule for KLUE-NLI dataset
"""
import os
import logging

from typing import Any, List, Union, Optional

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, ByT5Tokenizer)
from datasets import (DatasetDict, Dataset, load_dataset,
                      interleave_datasets, load_from_disk, concatenate_datasets)

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

    def predict_dataloader(self):
        # same as test_dataloader()
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)

    # dataset을 바로 노출하는 메서드
    def test_rawdataset(self):
        return self.dataset_test_iter

    def predict_rawdataset(self):
        return self.dataset_test_iter


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
            name="ynat", data_dir=basepath)

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

    def predict_dataloader(self):
        # same as test_dataloader()
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)

    # dataset을 바로 노출하는 메서드
    def test_rawdataset(self):
        return self.dataset_test_iter

    def predict_rawdataset(self):
        return self.dataset_test_iter


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

        # 데이터 특성 상, question/context+title을 모두 합쳐도 5200 bytes를 넘지 않는다.
        # max_seq_length가 5200이 넘어가면 별도의 sliding이 필요없다는 뜻.

        # split train into train/valid
        splitted_ds = klue_mrc_whole["train"].train_test_split(test_size=self.valid_proportion,)
        self.dataset_train_iter = splitted_ds["train"]
        self.dataset_valid_iter = splitted_ds["test"]

        # use validation dataset split as a test
        #self.dataset_test_iter = klue_mrc_whole["test"].shard(num_shards=100, index=99)
        self.dataset_test_iter = klue_mrc_whole["test"]

        def filter_longer_answer(example):
            # max()를 사용한 방법이 좋지는 않았음. EM ~49/ RW ~50
            #idx = example['answers']['text'].index(max(example['answers']['text'], key=len))
            #example['answers'] = {'text': example['answers']['text'][idx],
            #                      'start_idx': example['answers']['start_idx'][idx]}
            example['answers'] = {'text': example['answers']['text'][-1],
                                  'start_idx': example['answers']['start_idx'][-1]}
            return example

        # 정답이 여러개 인 것이 있을 수 있으므로, 하나로 줄여낸다.
        # torch.utils.data.DataLoader는 엘리먼트 수가 정렬되어 있어야 함.
        self.dataset_train_iter = self.dataset_train_iter.map(filter_longer_answer)
        self.dataset_valid_iter = self.dataset_valid_iter.map(filter_longer_answer)
        self.dataset_test_iter = self.dataset_test_iter.map(filter_longer_answer)

        # 일단은 정답이 없는게 너무 자주 나타나는 것도 문제가 있는 듯. 노출 정도를 줄여보자.
        # 심플하게, 정답이 있는 쪽을 1.5배로 oversampling 하도록 한다.
        have_anss = self.dataset_train_iter.filter(lambda example: example['plausible_answer'] is False)
        behalf_train = have_anss.shuffle(seed=99).shard(num_shards=2, index=0)
        self.dataset_train_iter = concatenate_datasets([self.dataset_train_iter, behalf_train])

        """
        # train은 데이터 보강이 필요하다: 정답이 context에 없을 경우에는 빈 값을 학습해야 함.
        # --> 2023.06.20. 1/5 증강에서도 너무 많이 틀리는 문제가 있다. 일단 off.
        have_anss = self.dataset_train_iter.filter(lambda example: example['plausible_answer'] is False)
        behalf_train = have_anss.shuffle(seed=99).shard(num_shards=5, index=0)

        def _generate_nonanswer(example):
            if example['plausible_answer'] is True:
                return example

            lbl = example['answers']['text']
            ctx_list = example['context'].split(". ")
            newctx = ''
            for part in ctx_list:
                if part.find(lbl) > -1:
                    continue
                newctx += part + ". "
            example['context'] = newctx
            example['answers'] = {'text': '[답 없음]', 'start_idx': -1}
            example['plausible_answer'] = True
            return example

        behalf_train = behalf_train.map(_generate_nonanswer)
        self.dataset_train_iter = concatenate_datasets([self.dataset_train_iter, behalf_train])
        """

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
