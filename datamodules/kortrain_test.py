"""
    Pytorch-lightning datamodule for Internal Text Dataset.
"""

import logging

from typing import Any, List, Union, Optional

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, ByT5Tokenizer)
from datasets import (DatasetDict, Dataset, load_dataset, load_from_disk,
                      interleave_datasets, concatenate_datasets)

class korTrainTextDataModule(pl.LightningDataModule):
    def __init__(self,
                 valid_proportion: float=0.05,
                 batch_size: int=32,
                 use_mtl: bool=False,
                 **kwargs):
        super(korTrainTextDataModule, self).__init__()
        self.valid_proportion = valid_proportion
        self.batch_size = batch_size
        self.use_mtl = use_mtl
        return

    def prepare_data(self):
        return

    def setup(self, stage: str=""):
        if self.use_mtl:
            whole_ds = load_from_disk('/home/jhshin/Works/ptlm-downstream-test/kortrain-classifer-test-230117/korail_cls_mtl/')
            #whole_ds = load_from_disk('/home/jhshin/Works/ptlm-downstream-test/kortrain-classifer-test-230117/korail_cls_mtl_maindataonly/')
        else:
            whole_ds = load_from_disk('/home/jhshin/Works/ptlm-downstream-test/kortrain-classifer-test-230117/korail_cls/')

        self.i2l = {idx: names for idx, names in enumerate(whole_ds['train'].features['label'].names)}
        self.l2i = {names: idx for idx, names in enumerate(whole_ds['train'].features['label'].names)}

        # FIXME: 부서서무|처리과서무 를 undersampling
        nonmj = whole_ds["train"].filter(lambda ex: ex["label"] != self.l2i['부서서무|처리과서무'])
        mj = whole_ds["train"].filter(lambda ex: ex["label"] == self.l2i['부서서무|처리과서무'])
        whole_ds["train"] = concatenate_datasets([nonmj, mj.shard(num_shards=7, index=1)])

        # split train into train/valid
        splitted_ds = whole_ds["train"].train_test_split(test_size=self.valid_proportion,
                                                         shuffle=True,
                                                         seed=123456,)
        #self.dataset_train_iter = splitted_ds["train"].shuffle(seed=98765).shard(num_shards=16, index=2)
        self.dataset_train_iter = splitted_ds["train"].shuffle(seed=76543).shard(num_shards=2, index=0)
        self.dataset_valid_iter = splitted_ds["test"]

        # use validation dataset split as a test
        self.dataset_test_iter = whole_ds["test"]

    def train_dataloader(self):
        return DataLoader(self.dataset_train_iter, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid_iter, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test_iter, batch_size=self.batch_size, num_workers=4)

    # id -> label. 반드시 korTrainTextDataModule().setup() 호출 후에 valid하다.
    # 그러므로, datacollator와 바인딩하기 위해서는 lazy init를 수행하는 id_to_label()을 쓸 것.
    def id_to_label_map_dict(self):
        return self.i2l

    # label -> id
    def label_to_id_map_dict(self):
        # use generator expression for lazy evaluation
        return self.l2i

    def id_to_label_func(self):
        #return lambda x: self.dataset_train_iter.features['label'].names[x]
        return lambda x: self.i2l[x]

    def label_to_id_func(self):
        return lambda x: self.l2i[x]
