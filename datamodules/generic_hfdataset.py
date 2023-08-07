"""
    Huggingface dataset을 사용하는 data loader.

    Copyright (C) 2023~ Jong-hun Shin. ETRI LIRS.

    load_from_disk()를 사용하여 로딩 가능한 데이터셋에, 지정된 context(text)와
    label feature name을 지정받아, 이를 읽어들이는 간단한 형태의 dataloader.
"""
import re

from typing import Any, List, Union, Optional, Callable
from functools import partial

import pytorch_lightning as pl
import datasets

from torch.utils.data import DataLoader


class GenericHFDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int,
                 train_files: Union[str, List[str]],
                 valid_files: Optional[Union[str, List[str]]]=None,
                 test_files: Optional[Union[str, List[str]]]=None,
                 valid_proportions: float=0.05,
                 test_proportions: float=0.0,
                 max_seq_length: Optional[int]=None,
                 tokenizer: Optional[Callable]=None,
                 do_truncate: bool=False,
                 hf_cache_dir: Optional[str]=None,
                 **kwargs):
        """
        입력:
        train_files: 학습 데이터, 파일 목록을 넣거나 파일 이름을 전달 (필수)
        valid_files: 검증 데이터, 파일 목록을 넣거나 파일 이름을 전달 (선택)
        test_files: 테스트 데이터, 파일 목록을 넣거나 파일 이름을 전달 (선택)

        valid_proportions: valid_files가 None일 때, train_files로 부터 분할 할 크기.
        검증 데이터가 None이 아닌 경우, 해당 값은 무시됨. 기본값: 5%
        test_proportions: test_files가 None일 때, train_files로 부터 분할 될 크기.
        테스트 데이터가 None이 아닌 경우, 해당 값을 무시됨. 기본값: 0%

        max_seq_length: 길이 제한이 필요할 경우, tokenizer를 미리 구동하여 길이를 제한한다.
        tokenizer: max_seq_length > 0 일 때, 있어야 길이를 제한함. 그렇지 않을 경우, 문자의 길이로 제한.
        do_truncate: False일 경우, max_seq_length보다 클 경우 엘리먼트를 제외, True일 경우, 단순 truncate.
        hf_cache_dir: 중간 편집이 있을 경우, hf datasets의 cache dir 위치.
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_files = [train_files] if isinstance(train_files, str) else train_files
        self.valid_files = [valid_files] if isinstance(valid_files, str) else valid_files
        self.test_files = [test_files] if isinstance(test_files, str) else test_files
        self.valid_props = valid_proportions
        self.test_props = test_proportions

        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.do_truncate = do_truncate
        self.hf_cache_dir = hf_cache_dir

        return

    def prepare_data(self):
        # do nothing.
        return

    def _get_dataset_from_disk(self, dsets: Union[str, list[str]]):
        if dsets is None:
            return None

        if isinstance(dsets, str):
            # get sharding information
            shard_val = 0
            sh_srch = re.search(r':([0-9]*)$', dsets)
            if sh_srch is not None and len(sh_srch.groups()) > 0:
                shard_val = int(sh_srch.groups()[0])
                dsets = dsets[:-(len(sh_srch.groups()[0])+1)]
                print(f"** Sharding suffix detected: 1/{shard_val} of "
                      f"data samples will be used in {dsets}")

            ads = datasets.load_from_disk(dsets)
            if shard_val > 0:
                ads = ads.shuffle()
                ads = ads.shard(num_shards=shard_val, index=0)
            return ads
        elif isinstance(dsets, list):
            dss = []
            for dset in dsets:
                # get sharding information
                shard_val = 0
                sh_srch = re.search(r':([0-9]*)$', dset)
                if sh_srch is not None and len(sh_srch.groups()) > 0:
                    shard_val = int(sh_srch.groups()[0])
                    dset = dset[:-(len(sh_srch.groups()[0])+1)]
                    print(f"** Sharding suffix detected: 1/{shard_val} of "
                          f"data samples will be used in {dset}")
                ads = datasets.load_from_disk(dset)
                if shard_val > 0:
                    ads = ads.shuffle()
                    ads = ads.shard(num_shards=shard_val, index=0)
                dss.append(ads)

            return datasets.concatenate_datasets(dss)
        else:
            raise NotImplementedError("generic_hfdataset: must be str, or list[str].")

    def setup(self, stage: str=""):
        # list of datasets.Dataset
        self.dataset_test_iter = self._get_dataset_from_disk(self.test_files)
        self.dataset_valid_iter = self._get_dataset_from_disk(self.valid_files)
        self.dataset_train_iter = self._get_dataset_from_disk(self.train_files)

        if self.test_props > 0.0 and self.dataset_test_iter is None:
           tt_split = self.dataset_train_iter.train_test_split(test_size=self.test_props, shuffle=True,
                                                               seed=592821,)
           self.dataset_train_iter = tt_split["train"]
           self.dataset_test_iter = tt_split["test"]

        if self.valid_props > 0.0 and self.dataset_valid_iter is None:
            tv_split = self.dataset_train_iter.train_test_split(test_size=self.valid_props, shuffle=True,
                                                                seed=395810,)
            self.dataset_train_iter = tv_split["train"]
            self.dataset_valid_iter = tv_split["test"]

        # do_truncate == True 일 경우에는 tokenizer에서 제한함.
        if self.max_seq_length is not None and self.max_seq_length > 0 and self.do_truncate is False:
            print(f"max sequence length: {self.max_seq_length} > 0, now starts filtering(TO DISCARD) by length.")
            def check_length_func(x, tokenizer, limit):
                if tokenizer is None:
                    return sum([len(y) for y in x.values()]) < limit
                else:
                    return sum([len(tokenizer(y)["input_ids"]) for y in x.values()]) < limit

            func = partial(check_length_func, tokenizer=self.tokenizer, limit=self.max_seq_length)

            # discard, we will use filter()
            if self.dataset_train_iter is not None:
                self.dataset_train_iter = self.dataset_train_iter.filter(func, num_proc=2,)
            if self.dataset_valid_iter is not None:
                self.dataset_valid_iter = self.dataset_valid_iter.filter(func, num_proc=2,)

        print(self.dataset_train_iter)
        print(self.dataset_valid_iter)

        return

    def save_to_disk(self, save_path):
        dsd = {}
        if self.dataset_train_iter is None:
            self.setup()

        if self.dataset_train_iter is not None:
            dsd["train"] = self.dataset_train_iter
        if self.dataset_valid_iter is not None:
            dsd["validation"] = self.dataset_valid_iter
        if self.dataset_test_iter is not None:
            dsd["test"] = self.dataset_test_iter
        v = datasets.DatasetDict(dsd)

        v.save_to_disk(save_path)
        return

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
