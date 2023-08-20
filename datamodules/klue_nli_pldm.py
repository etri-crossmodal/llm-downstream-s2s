
"""
    Pytorch-lightning datamodule for KLUE-NLI dataset
"""
import os
import logging

from copy import deepcopy
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
        #splitted_ds = klue_nli_whole["train"].train_test_split(test_size=self.valid_proportion,)
        #self.dataset_train_iter = splitted_ds["train"]
        #self.dataset_valid_iter = splitted_ds["test"]

        # 임시..
        self.dataset_train_iter = klue_nli_whole["train"]
        self.dataset_valid_iter = klue_nli_whole["test"]

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
        #splitted_ds = klue_mrc_whole["train"].train_test_split(test_size=self.valid_proportion,)
        #self.dataset_train_iter = splitted_ds["train"]
        #self.dataset_valid_iter = splitted_ds["test"]
        self.dataset_train_iter = klue_mrc_whole["train"]
        self.dataset_valid_iter = klue_mrc_whole["test"]

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

        # 반대로 정답이 없는것을 더 많이 나타내게 한다. collator에서 보정할 기회를 줄 것이다.
        no_anss = self.dataset_train_iter.filter(lambda example: example['plausible_answer'] is True)
        behalf_train = no_anss.shuffle(seed=99).shard(num_shards=2, index=0)
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


class KLUENERDataModule(pl.LightningDataModule):
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
        klue_ner_whole = load_dataset(
            path=os.path.join(basepath, "klue_data.py"),
            name="ner", data_dir=basepath)

        # tagged_sent 컬럼을 지우든지, 아니면 이걸 지우든지 해야 함. list - str 엘리먼트 수가 달라서
        # 발생하는 문제임.
        klue_ner_whole['test'] = klue_ner_whole['test'].remove_columns(["sentence", "labels",])
        klue_ner_whole['train'] = klue_ner_whole['train'].remove_columns(["sentence", "labels",])

        # split train into train/valid
        #splitted_ds = klue_nli_whole["train"].train_test_split(test_size=self.valid_proportion,)
        #self.dataset_train_iter = splitted_ds["train"]
        #self.dataset_valid_iter = splitted_ds["test"]

        # 임시..
        self.dataset_train_iter = klue_ner_whole["train"]
        self.dataset_valid_iter = klue_ner_whole["test"]

        # use validation dataset split as a test
        self.dataset_test_iter = klue_ner_whole["test"]

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



class KLUEDPDataModule(pl.LightningDataModule):
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
        basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "klue_datasets/")
        klue_dp_whole = load_dataset(
            path=os.path.join(basepath, "klue_data.py"),
            name="dp_hfstyle", data_dir=basepath)

        def _gen_dp_pred_string(example):
            wip_str = ''
            for w, l, i, p in zip(example['word_form'], example['lemma'], example['index'], example['pos']):
                ls = l.split(' ')
                ps = p.split('+')
                # w/first-last feature
                #wip_str += f"{i}/{w}/{l}/{p}/{ls[0]}:{ps[0]}" + '/' + ("NONE" if len(ps)==1 else f"{ls[-1]}:{ps[-1]}") + '\n'
                # simplified, v3a, v3b
                # wip_str += f"{i}/{w}/{l}/{p}\n"
                # v3c
                #wip_str += f"{i}/{w}/{len(w)}/{l}/{p}\n"

                # v3d, append start-end lemma and pos
                wip_str += f"{i}/{w}/{len(w)}/{l}/{p}/{ls[0]}:{ps[0]}" + '/' + ("NONE" if len(ps)==1 else f"{ls[-1]}:{ps[-1]}") + '\n'

            #lip_str = '▁'.join([f"([{l}], {i}, {p})" 
            #    for l, i, p in zip(example['lemma'], example['index'], example['pos'])])
            #lhdr_str = '▁'.join([f"([{l}], {h}, {d})" 
            #    for l, h, d in zip(example['lemma'], example['head'], example['deprel'])])
            # splitter를 reserved token으로 적용 시
            # v3, v3a
            #wdr_str = '▁'.join([f"({w}<extra_id_0>{i}, {h}, {d})" 
            #    for w, i, h, d in zip(example['word_form'], example['index'], example['head'], example['deprel'])])
            wdr_counts = len(example['word_form'])

            # v3b
            #ihd_str = '▁'.join([f"({i}, {h}, {d})" 
            #    for i, h, d in zip(example['index'], example['head'], example['deprel'])])
            # v3c, v3d
            ilhd_str = '▁'.join([f"({i}/{len(w)}, {h}, {d})" 
                for i, w, h, d in zip(example['index'], example['word_form'], example['head'], example['deprel'])])

            # v3e -> v3d에 비해 성능이 낮음
            #ilphd_str = '▁'.join([f"({i}/{len(w)}, {h}, {p.split('+')[0][:2]}, {d})" 
            #    for i, w, p, h, d in zip(example['index'], example['word_form'], example['pos'],
            #                             example['head'], example['deprel'])])

            # ver 1
            #return { 'label': f"lemma: {lip_str}\ndeprel: {lhdr_str}" }

            # ver 2
            #return { 'ma_out': lip_str, 'label': f"deprel: {lhdr_str}" }

            # ver 3
            #return { 'ma_out': wip_str, 'label': f"deprel: {wdr_str}" }

            # ver 3a
            #return { 'ma_out': wip_str, 'label': f"word_counts:{wdr_counts}\ndeprel: {wdr_str}" }

            # ver 3b: word_form을 중복하게하여 혼동하지 않도록, 그리고 word counts를 마지막에 출력하도록.
            #return { 'ma_out': wip_str, 'label': f"deprel: {ihd_str}\nword_counts: {wdr_counts}" }

            # ver 3c(new) 3b + ihd_str에서 index가 가리키는 word의 word_form char 길이가 얼마인지를 예측하게 함
            # v3d에서도 사용.
            return { 'ma_out': wip_str, 'label': f"deprel: {ilhd_str}\nword_counts: {wdr_counts}" }

            # v3d에서는 적용하지 않지만, 예측에서 first or last pos를 예측하게 하는게 더 좋지 않을까 생각 중.
            # v3e 에서 먼저 p[0]를 예측. -> 실패. pos를 예측하는 것은, 2글자로 줄이는 것도 충분치 않고, 오류만 늘음
            # return { 'ma_out': wip_str, 'label': f"deprel: {ilphd_str}\nword_counts: {wdr_counts}" }
            

        # split train into train/valid
        #splitted_ds = klue_dp_whole["train"].train_test_split(test_size=self.valid_proportion,)
        #self.dataset_train_iter = splitted_ds["train"]
        #self.dataset_valid_iter = splitted_ds["test"]

        # 임시..
        self.dataset_train_iter = klue_dp_whole["train"]
        self.dataset_valid_iter = deepcopy(klue_dp_whole["test"])

        # use validation dataset split as a test
        self.dataset_test_iter = klue_dp_whole["test"]

        # 전처리
        self.dataset_train_iter = self.dataset_train_iter.map(_gen_dp_pred_string)
        self.dataset_valid_iter = self.dataset_valid_iter.map(_gen_dp_pred_string)
        self.dataset_test_iter = self.dataset_test_iter.map(_gen_dp_pred_string)

        # 컬럼 정렬을 위해 입/출력만 남김
        self.dataset_train_iter = self.dataset_train_iter.remove_columns(['index', 'word_form', 'lemma',
                                                                          'pos', 'head', 'deprel'])
        self.dataset_valid_iter = self.dataset_valid_iter.remove_columns(['index', 'word_form', 'lemma',
                                                                          'pos', 'head', 'deprel'])
        self.dataset_test_iter = self.dataset_test_iter.remove_columns(['index', 'word_form', 'lemma',
                                                                        'pos', 'head', 'deprel'])

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
