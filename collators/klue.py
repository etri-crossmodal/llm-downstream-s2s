"""
    Data Collator of a Enc-Dec Structure NNs, for KLUE datasets.

    Copyright (C) 2023~, Jong-hun Shin. ETRI LIRS.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Union, Optional, Dict

from transformers import AutoTokenizer, BatchEncoding

@dataclass
class KLUENLIDataCollator:
    """
        KLUE-NLI Data Collator.
    """
    # dataclass definitions
    premise_field_name: str="premise"
    hypothesis_field_name: str="hypothesis"
    label_field_name: str="label"
    tokenizer: Optional[Callable]=field(default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    # 0: 함의/entailment, 1: 중립/neutral, 2: 모순/contradiction
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=field(default_factory=lambda: {0:'entailment', 1:'neutral', 2:'contradiction'})

    def __call__(self, examples: dict[str, Any]) -> dict[str, Any]:
        """
        출력 형식 - Dict, { 'input_ids': ndarray-like, 'attention_mask': ndarray-like, 'labels': ndarray-like }
        """
        if isinstance(examples, dict):
            premises = examples[self.premise_field_name]
            hypothesises = examples[self.hypothesis_field_name]
            labels = examples[self.label_field_name]

            input_texts = []
            for idx, premise in enumerate(premises):
                hyp = hypothesises[idx]
                #input_texts.append(f"다음 전제-가설간의 관계가 함의면 'entailment', 중립이면 'neutral', 모순이면 'contradiction'를 출력해라:\n\n전제: {premise}\n가설: {hyp}")
                input_texts.append(f"KLUE NLI task - premise: [{premise}], hypothesis: [{hyp}]\n")

            if isinstance(self.label_map, dict):
                # label-text mapper via dict.
                label_texts = [ self.label_map[x.item()] for x in labels]
            elif isinstance(self.label_map, Callable):
                label_texts = [ self.label_map(x.item()) for x in labels]
            else:
                raise NotImplementedError

            return BatchEncoding(self.tokenizer(text=input_texts, text_target=label_texts, padding='longest', return_tensors="pt"))
        else:
            raise NotImplementedError

        return None


@dataclass
class KLUEYNATDataCollator:
    """
        KLUE-YNAT Data Collator.
        테스트용. 원래는 기사도 가져와서 결정해야 함
    """
    # dataclass definitions
    title_field_name: str="title"
    label_field_name: str="label"
    tokenizer: Optional[Callable]=field(default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=field(default_factory=lambda: {0:'IT/science', 1:'economy',
        2:'social', 3:'life and culture', 4:'world', 5:'sports', 6:'politics'})

    def __call__(self, examples: dict[str, Any]) -> dict[str, Any]:
        if isinstance(examples, dict):
            titles = examples[self.title_field_name]
            labels = examples[self.label_field_name]

            input_texts = []
            for idx, title in enumerate(titles):
                input_texts.append(f"KLUE YNAT task: {title}\n")

            if isinstance(self.label_map, dict):
                # label-text mapper via dict.
                label_texts = [self.label_map[x.item()] for x in labels]
            elif isinstance(self.label_map, Callable):
                label_texts = [self.label_map(x.item()) for x in labels]
            else:
                raise NotImplementedError

            return BatchEncoding(self.tokenizer(text=input_texts, text_target=label_texts, padding='longest', return_tensors="pt"))
        else:
            raise NotImplementedError

        return None
