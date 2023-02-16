"""
    task 선택/지원을 위한 유틸리티.

    Copyright (C) 2023, Jong-hun Shin. ETRI LIRS.
"""
import jellyfish

from collections import Counter

from transformers import AutoTokenizer

from datamodules.nsmc_pldm import NSMCDataModule
from datamodules.klue_nli_pldm import KLUENLIDataModule, KLUEYNATDataModule
from datamodules.kornli_pldm import KorNLIDataModule
from datamodules.pawsx_pldm import paws_xDataModule
from datamodules.kortrain_test import korTrainTextDataModule

from collators import (generic, klue, pawsx, korail_internal)


def get_task_data(task_name: str, batch_size: int, tokenizer_str: str):
    """
    태스크에 맞는 lightning datamodule과, collator를 반환합니다.
    collator는 generic collator를 응용하거나, collators/ 디렉터리 아래에
    파일을 추가합니다.

    반환 값: data_module, collator, label-id map for corrector
    """
    data_module, collator = None, None
    gold_labels = None

    if task_name == "nsmc-naive":
        # NSMC - naive version
        data_module = NSMCDataModule(batch_size=batch_size)
        collator = generic.GenericDataCollator(input_field_name="document",
                                               label_field_name="label",
                                               tokenizer=AutoTokenizer.from_pretrained(tokenizer_str),
                                               label_map={0: 'positive', 1: 'negative'})
        gold_labels = {"positive":0, "negative":1}
    elif task_name == "nsmc-prompted":
        data_module = NSMCDataModule(batch_size=batch_size)
        collator = generic.GenericPromptedDataCollator(input_field_name="document",
                label_field_name="label",
                input_template="nsmc sentiment classification: {{ input }}",
                label_template="{{ label }}",
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_str),
                label_map={0:'positive', 1:'negative'})
        gold_labels = {"positive":0, "negative":1}
    elif task_name == "klue-nli-prompted":
        # Example 2: KLUE-NLI
        data_module = KLUENLIDataModule(batch_size=batch_size)
        collator = klue.KLUENLIDataCollator(tokenizer=AutoTokenizer.from_pretrained(tokenizer_str),)
        gold_labels = {"entailment":0, "neutral":1, "contradiction":2}
    elif task_name == "klue-ynat-prompted":
        # Example: KLUE-YNAT
        data_module = KLUEYNATDataModule(batch_size=batch_size)
        collator = klue.KLUEYNATDataCollator(tokenizer=AutoTokenizer.from_pretrained(tokenizer_str),)
        gold_labels = {"different":0, "paraphrase":1}
    elif task_name == 'kornli-prompted':
        # Example 3: KorNLI
        data_module = KorNLIDataModule(batch_size=batch_size)
        collator = klue.KLUENLIDataCollator(tokenizer=AutoTokenizer.from_pretrained(tokenizer_str),)
        gold_labels = {"entailment":0, "neutral":1, "contradiction":2}
    elif task_name == 'paws-x-kor':
        data_module = paws_xDataModule(batch_size=batch_size)
        collator = pawsx.PAWS_XDataCollator(tokenizer=AutoTokenizer.from_pretrained(tokenizer_str),)
        gold_labels = {'IT/science':0, 'economy':1, 'social':2,
                       'life and culture':3, 'world':4, 'sports':5, 'politics':6}
    elif task_name == 'kr-internal':
        # Korail, Internal Dataset, Multiclass classification problem.
        #data_module = korTrainTextDataModule(batch_size=batch_size)
        # MTL
        data_module = korTrainTextDataModule(batch_size=batch_size, use_mtl=False)
        collator = korail_internal.korailCollatorV1(
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_str),
            label_map=data_module.id_to_label_func(),
            length_limit=512)
        """
        collator = generic.GenericPromptedDataCollator(input_field_name="title",
                label_field_name="label",
                input_template="Classification Test:\n\n{{ input }}",
                label_template="{{ label }}",
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_str),
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
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_str),
                # get callable to label mapping
                label_map=data_module.id_to_label_func(),
                #label_map=data_module.id_to_label_map_dict(),
                )
        # FIXME: 현재는 label을 얻기 위해 data_module.setup()을 호출해야 함.
        data_module.setup()
        gold_labels = data_module.label_to_id_map_dict()
    elif task_name == "ko-en-translate":
        # Example 3: Translation
        """
        data_module = ...
        collator = ...
        gold_labels = None
        """
        raise NotImplemented
    else:
        print('-task option now supported on: nsmc-naive, nsmc-prompted, klue-nli-prompted, klue-ynat-prompted, ...')
        raise NotImplemented

    return data_module, collator, gold_labels


def get_unique_labels(labels: list[str]):
    cnts = Counter(labels)
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
