"""
    task 선택/지원을 위한 유틸리티.

    Copyright (C) 2023, Jong-hun Shin. ETRI LIRS.
"""
import jellyfish

from typing import Optional, List, Union
from collections import Counter

from transformers import AutoTokenizer

from datamodules.nsmc_pldm import NSMCDataModule
from datamodules.klue_nli_pldm import (
        KLUENLIDataModule, KLUEYNATDataModule, KLUEMRCDataModule, KLUENERDataModule
        )
from datamodules.kornli_pldm import KorNLIDataModule
from datamodules.pawsx_pldm import paws_xDataModule
from datamodules.kortrain_test import korTrainTextDataModule
from datamodules.generic_tsv import GenericTSVDataModule
from datamodules.korquad_v1 import KorQuadV1DataModule
from datamodules.generic_hfdataset import GenericHFDataModule

from collators import (generic, klue, pawsx, korail_internal, korquad_v1)


def get_task_data(task_name: str, batch_size: int,
                  tokenizer_str: str,
                  train_data_file: Optional[List]=None,
                  valid_data_file: Optional[List]=None,
                  test_data_file: Optional[List]=None,
                  valid_proportions: float=0.0,
                  test_proportions: float=0.0,
                  max_seq_length: Optional[int]=None,
                  do_truncate: bool=False,
                  hf_cache_dir: Optional[str]=None):
    """
    태스크에 맞는 lightning datamodule과, collator를 반환합니다.
    collator는 generic collator를 응용하거나, collators/ 디렉터리 아래에
    파일을 추가합니다.

    반환 값: data_module, collator, label-id map for corrector
    """
    data_module, collator = None, None
    gold_labels = None

    if max_seq_length == 0:
        max_seq_length = None

    if task_name == "nsmc-naive":
        # NSMC - naive version
        data_module = NSMCDataModule(batch_size=batch_size)
        collator = generic.GenericDataCollator(input_field_name="document",
                                               label_field_name="label",
                                               tokenizer=AutoTokenizer.from_pretrained(tokenizer_str,
                                                                                       use_auth_token=True),
                                               label_map={0: 'positive', 1: 'negative'},
                                               max_seq_length=max_seq_length,)
        gold_labels = {"positive":0, "negative":1}
    elif task_name == "nsmc-prompted":
        data_module = NSMCDataModule(batch_size=batch_size)
        collator = generic.GenericPromptedDataCollator(
            input_field_name="document",
            label_field_name="label",
            input_template="nsmc sentiment classification: {{ input }}",
            label_template="{{ label }}",
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_str, use_auth_token=True),
            label_map={0:'positive', 1:'negative'},
            max_seq_length=max_seq_length,)
        gold_labels = {"positive":0, "negative":1}
    elif task_name == "klue-nli":
        # Example 2: KLUE-NLI
        base_tag_pair = {"<extra_id_0>":0, "<extra_id_1>":1, "<extra_id_2>":2}
        #base_tag_pair = {"함의":0, "중립":1, "모순":2}
        data_module = KLUENLIDataModule(batch_size=batch_size)
        collator = klue.KLUENLIDataCollator(tokenizer=AutoTokenizer.from_pretrained(
            tokenizer_str, use_auth_token=True),
            label_map=dict([(v, k) for k, v in base_tag_pair.items()]), # invert k-v pair
            max_seq_length=max_seq_length,)
        gold_labels = base_tag_pair
    elif task_name == "klue-ynat":
        # Example: KLUE-YNAT
        data_module = KLUEYNATDataModule(batch_size=batch_size)
        collator = klue.KLUEYNATDataCollator(tokenizer=AutoTokenizer.from_pretrained(tokenizer_str,
                                                                                     use_auth_token=True),
                                            max_seq_length=max_seq_length,)
        gold_labels = {'IT과학':0, '경제':1, '사회':2, '생활문화':3, '세계':4, '스포츠':5, '정치':6}
    elif task_name == 'kornli-prompted':
        # Example 3: KorNLI
        data_module = KorNLIDataModule(batch_size=batch_size)
        collator = klue.KLUENLIDataCollator(tokenizer=AutoTokenizer.from_pretrained(tokenizer_str,
                                                                                    use_auth_token=True),
                                            max_seq_length=max_seq_length,)
        gold_labels = {"entailment":0, "neutral":1, "contradiction":2}
    elif task_name == 'paws-x-kor':
        data_module = paws_xDataModule(batch_size=batch_size)
        collator = pawsx.PAWS_XDataCollator(tokenizer=AutoTokenizer.from_pretrained(tokenizer_str,
                                                                                    use_auth_token=True),
                                            max_seq_length=max_seq_length,)
        gold_labels = {'IT/science':0, 'economy':1, 'social':2,
                       'life and culture':3, 'world':4, 'sports':5, 'politics':6}
    elif task_name == 'kr-internal':
        # Korail, Internal Dataset, Multiclass classification problem.
        #data_module = korTrainTextDataModule(batch_size=batch_size)
        # MTL
        data_module = korTrainTextDataModule(batch_size=batch_size, use_mtl=False)
        collator = korail_internal.korailCollatorV1(
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_str, use_auth_token=True),
            label_map=data_module.id_to_label_func(),
            length_limit=max_seq_length)
        """
        collator = generic.GenericPromptedDataCollator(input_field_name="title",
                label_field_name="label",
                input_template="Classification Test:\n\n{{ input }}",
                label_template="{{ label }}",
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_str, use_auth_token=True),
                # get callable to label mapping
                label_map=data_module.id_to_label_func(),
                #label_map=data_module.id_to_label_map_dict(),
                )
        """
        # FIXME: 현재는 label을 얻기 위해 data_module.setup()을 호출해야 함.
        data_module.setup()
        gold_labels = data_module.label_to_id_map_dict()
    elif task_name == 'kr-internal-mtl':
        # Korail, Internal Dataset, Multiclass classification problem.
        #data_module = korTrainTextDataModule(batch_size=batch_size)
        # MTL
        data_module = korTrainTextDataModule(batch_size=batch_size, use_mtl=True)
        collator = generic.GenericPromptedDataCollator(input_field_name="title",
                label_field_name="label",
                input_template="Multitask classification:\n\n{{ input }}",
                label_template="{{ label }}",
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_str, use_auth_token=True),
                # get callable to label mapping
                label_map=data_module.id_to_label_func(),
                #label_map=data_module.id_to_label_map_dict(),
                max_seq_length=max_seq_length,
                )
        # FIXME: 현재는 label을 얻기 위해 data_module.setup()을 호출해야 함.
        data_module.setup()
        gold_labels = data_module.label_to_id_map_dict()
    elif task_name == 'klue-mrc':
        # klue_datamodules에 포함된 데이터셋을 사용한 학습 및 평가 체계 예시.
        # data module만 별도로 사용.
        data_module = KLUEMRCDataModule(valid_proportions=0.05,
                                        batch_size=batch_size)
        collator = klue.KLUEMRCDataCollator(tokenizer=AutoTokenizer.from_pretrained(tokenizer_str,
                                                                                    use_auth_token=True),
                                            label_map=None,
                                            max_seq_length=max_seq_length,)
        gold_labels = None
    elif task_name == 'klue-ner':
        # klue_datamodules에 포함된 데이터셋을 사용한 학습 및 평가 체계 예시.
        # data module만 별도로 사용.
        data_module = KLUENERDataModule(valid_proportions=0.05,
                                        batch_size=batch_size)
        collator = klue.KLUENERDataCollator(tokenizer=AutoTokenizer.from_pretrained(tokenizer_str,
                                                                                    use_auth_token=True),
                                            label_map=None,
                                            max_seq_length=max_seq_length,)
        gold_labels = None
    elif task_name == 'korquad-v1':
        data_module = KorQuadV1DataModule(valid_proportions=0.05, batch_size=batch_size)
        collator = korquad_v1.KorQuadV1DataCollator(
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_str, use_auth_token=True),
                label_map=None, max_seq_length=max_seq_length,)
        gold_labels = None
    elif task_name == 'hfdataset':
        # huggingface dataset에서 제공하는 text-label pair dataset
        data_module = GenericHFDataModule(batch_size, train_data_file, valid_data_file, test_data_file,
                                          valid_proportions, test_proportions, max_seq_length,
                                          AutoTokenizer.from_pretrained(tokenizer_str,
                                                                        use_auth_token=True),
                                          do_truncate,
                                          hf_cache_dir=hf_cache_dir)
        collator = generic.GenericDataCollator(input_field_name="text",
                                               label_field_name="label",
                                               tokenizer=AutoTokenizer.from_pretrained(tokenizer_str,
                                                                                       use_auth_token=True),
                                               label_map=None,
                                               max_seq_length=max_seq_length,)
        gold_labels = None
    else:
        # generic supervised seq2seq training, with -train_data, -valid_data, -test_data option.
        data_module = GenericTSVDataModule(batch_size, train_data_file, valid_data_file, test_data_file,
                                           ["text", "target_text"], "\t",
                                           valid_proportions, test_proportions,
                                           max_seq_length,
                                           AutoTokenizer.from_pretrained(tokenizer_str,
                                                                         use_auth_token=True),
                                           do_truncate,
                                           hf_cache_dir=hf_cache_dir)
        collator = generic.GenericDataCollator(input_field_name="text",
                                               label_field_name="target_text",
                                               tokenizer=AutoTokenizer.from_pretrained(tokenizer_str,
                                                                                       use_auth_token=True),
                                               label_map=None,
                                               max_seq_length=max_seq_length,)
        gold_labels = None

    return data_module, collator, gold_labels


def get_unique_labels(labels: Union[List[str], List[List[str]]]):

    if isinstance(labels[0], str):
        cnts = Counter(labels)
    elif isinstance(labels[0], list):
        # flatten
        cnts = Counter([item for sublist in labels for item in sublist])
    return cnts


def get_mislabel_correction_map(gold_labels, pred_labels_uniq):
    """
    edit distance를 사용해서 mislabeled prediction을 정정하는 mapping table을 생성한다.
    """
    correction_map = {}
    for k in pred_labels_uniq.keys():
        minimum_dist = 10000
        lowest_dist_lbl = ''
        for c in gold_labels.keys():
            ck_dist = jellyfish.levenshtein_distance(c, k)
            if minimum_dist > ck_dist:
                lowest_dist_lbl = c
                minimum_dist = ck_dist
        if k != lowest_dist_lbl:
            correction_map[k] = lowest_dist_lbl
    return correction_map
