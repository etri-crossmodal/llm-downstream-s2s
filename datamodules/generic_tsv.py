"""
    TSV(Tab-separated Values) 파일들을 받아서 입/출력으로 mapping하는
    generic datamodule.
    huggingface csv generic loader를 사용하지 않고 구현.
    hf csv generic loader는 pandas+apache arrow를 backend로 두고 파싱하는데,
    일부 tsv를 delimiter만 tab으로 바꾸고 csv reader로 읽었을 때
    잘 정제되지 않은 데이터에서 double quots 관련 오류가 발생할 여지가 있기 때문이다.

    column-key mapping을 할 경우 여러 컬럼을 입력으로 받을 수 있고,
    지정하지 않을 경우에는 "text"-"target_text" column을 갖도록 dataset을 반환함.

    Copyright (C) 2023~ Jong-hun Shin. ETRI LIRS.
"""
import logging

from pathlib import Path
from typing import Any, List, Union, Optional, Callable
from functools import partial

import tqdm
import pytorch_lightning as pl
import pandas as pd
import datasets

from torch.utils.data import DataLoader
#from transformers import (AutoTokenizer, ByT5Tokenizer)


class GenericTSVDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int,
                 train_files: Union[str, List[str]],
                 valid_files: Optional[Union[str, List[str]]]=None,
                 test_files: Optional[Union[str, List[str]]]=None,
                 column_map: Optional[List[str]]=None,
                 delimiter: str="\t",
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
        column_map: 컬럼 번호/이름을 매핑한다. 없을 경우, 첫번째 컬럼을 input_text로,
        두번째 컬럼을 label로 지정한다.
        예) column_map=["input_text", "label"]
        delimiter: "\t" tab 문자를 기본으로 한다.

        valid_proportions: valid_files가 None일 때, train_files로 부터 분할 할 크기.
        검증 데이터가 None이 아닌 경우, 해당 값은 무시됨. 기본값: 5%
        test_proportions: test_files가 None일 때, train_files로 부터 분할 될 크기.
        테스트 데이터가 None이 아닌 경우, 해당 값을 무시됨. 기본값: 10%

        max_seq_length: 길이 제한이 필요할 경우, tokenizer를 미리 구동하여 길이를 제한한다.
        tokenizer: max_seq_length > 0 일 때, 있어야 길이를 제한함. 그렇지 않을 경우, 문자의 길이로 제한.
        """
        super(GenericTSVDataModule, self).__init__()
        self.batch_size = batch_size
        self.train_files = [train_files] if isinstance(train_files, str) else train_files
        self.valid_files = [valid_files] if isinstance(valid_files, str) else valid_files
        self.test_files = [test_files] if isinstance(test_files, str) else test_files
        self.column_map = ["text", "target_text"] if column_map is None else column_map
        self.delimiter = delimiter
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

    def _get_dataset_from_files(self, filelist: List[str]):
        """
        파일 인자 목록으로 부터 datasets 인스턴스를 생성.
        """
        dss = []
        column_len = len(self.column_map)

        if filelist is not None:
            for a_file in tqdm.tqdm(filelist):
                # init
                datadict = {}
                for clid in range(column_len):
                    datadict[self.column_map[clid]] = []

                pif = Path(a_file)
                if pif.exists() and pif.is_file():
                    with open(a_file, 'rt') as in_f:
                        linecnt = 0
                        for aline in in_f:
                            alist = aline.strip().split(self.delimiter)
                            if len(alist) == column_len:
                                for clid, data in enumerate(alist):
                                    datadict[self.column_map[clid]].append(data)
                            else:
                                print(f"WARNING: in {a_file}, invalid line found. "
                                      f"skip line: #{linecnt+1}, {len(alist)-1} delimiter found.")
                            linecnt += 1
                    dss.append(datasets.Dataset.from_dict(datadict))
                elif pif.exists() and pif.is_dir():
                    # open with load_from_disk()
                    print("trying to load with datasets.load_from_disk()")
                    dsi = datasets.load_from_disk(a_file)
                    if isinstance(dsi, datasets.DatasetDict):
                        for dselem in dsi.values():
                            dss.append(dselem)
                    elif isinstance(dsi, datasets.Dataset):
                        dss.append(dsi)
                else:
                    # 수정 필요: 분명히 column name이 다를텐데...
                    print("path doesn't exists, so trying to load with datasets.load_dataset()")
                    dsi = datasets.load_dataset(a_file, cache_dir=self.hf_cache_dir)
                    if isinstance(dsi, datasets.DatasetDict):
                        for dselem in dsi.values():
                            dss.append(dselem)
                    elif isinstance(dsi, datasets.Dataset):
                        dss.append(dsi)
        else:
            return None

        if len(dss) <= 0:
            print("FATAL: filelist is available, but tsv/txt files were not processed properly.")
            print(f"FileList: {filelist}")
            return None

        return datasets.concatenate_datasets(dss)

    def setup(self, stage: str=""):
        # list of datasets.Dataset
        self.dataset_test_iter = self._get_dataset_from_files(self.test_files)
        self.dataset_valid_iter = self._get_dataset_from_files(self.valid_files)
        self.dataset_train_iter = self._get_dataset_from_files(self.train_files)

        if self.test_props > 0.0 and self.dataset_test_iter is None:
           tt_split = self.dataset_train_iter.train_test_split(test_size=self.test_props, shuffle=True,
                                                               seed=123456,)
           self.dataset_train_iter = tt_split["train"]
           self.dataset_test_iter = tt_split["test"]

        if self.valid_props > 0.0 and self.dataset_valid_iter is None:
            tv_split = self.dataset_train_iter.train_test_split(test_size=self.valid_props, shuffle=True,
                                                                seed=123456,)
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
            if self.dataset_test_iter is not None:
                self.dataset_test_iter = self.dataset_test_iter.filter(func, num_proc=2,)
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
