"""
    Generic Data Collator for Multi-class classification.

    Copyright (C) 2023~, Jong-hun Shin. ETRI LIRS.
"""
import jinja2

from dataclasses import dataclass, field
from typing import Any, Callable, Union, Optional, Dict

from transformers import AutoTokenizer, ByT5Tokenizer, BatchEncoding


_JENV = jinja2.Environment()


@dataclass
class GenericDataCollator:
    """
        Generic (Supervised) Data Collator. tokenizer를 입력받아, dataset의 label을 긍정/부정으로 부여하여
        T5/BART와 같은 Enc-Dec 모델의 입출력으로 생성한다.

        프롬프트 되지 않고, 입력에는 문장을, 출력에는 지정된 label로 치환한다.
    """
    # dataclass definitions
    input_field_name: str
    label_field_name: str
    tokenizer: Optional[Callable]=field(
        default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    label_map: Union[Dict[Any, str], Callable]=field(default_factory=lambda: { 0: "긍정", 1: "부정" })
    max_seq_length: Optional[int]=None

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        출력 형식 - Dict, { 'input_ids': ndarray-like, 'attention_mask': ndarray-like,
                            'labels': ndarray-like
                          }
        """
        if isinstance(examples, dict):
            # 보통의 DataLoader() iterator는 Dict[str, list[Any]]를 전달한다
            input_text = examples[self.input_field_name]
            labels = examples[self.label_field_name]

            if isinstance(self.label_map, dict):
                # label-text mapper via dict.
                label_texts = [self.label_map[x.item()] for x in labels]
            elif isinstance(self.label_map, Callable):
                label_texts = [self.label_map(x.item()) for x in labels]
            else:
                # use as-is
                label_texts = labels

            return BatchEncoding(self.tokenizer(text=input_text, text_target=label_texts,
                                                padding='longest', truncation="only_first",
                                                max_length=self.max_seq_length,
                                                return_tensors="pt", return_attention_mask=True,)
                                 )
        else:
            raise NotImplementedError

        return None


@dataclass
class GenericPromptedDataCollator:
    """
        Template 기반의 Generic/Supervised Data Collator.
        jinja2 template을 받아 데이터 샘플을 생성한다.

        FIXME: Prompt에 example을 복수개로 넣을 수 있도록 지원해야 함
    """
    input_field_name: str
    label_field_name: str
    input_template: str="아래 문장에서 나타나는 감정을 '긍정' 또는 '부정'으로 분류하시오:\n\n{{ input }}"
    label_template: str="(긍정/부정) => {{ label }}"
    examples_field_name: Optional[str]=None
    tokenizer: Optional[Callable]=field(
        default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small"))
    label_map: Optional[Union[Dict[Any, str], Callable]]=field(default_factory=lambda: { 0: "긍정", 1: "부정" })
    max_seq_length: Optional[int]=None

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        출력 형식 - Dict, { 'input_ids': ndarray-like, 'attention_mask': ndarray-like, 'labels': ndarray-like }
        """
        if isinstance(examples, dict):
            # 보통의 DataLoader() iterator는 Dict[str, list[Any]]를 전달한다
            input_text = examples[self.input_field_name]
            labels = examples[self.label_field_name]

            ipt_template = _JENV.from_string(self.input_template)
            lbl_template = _JENV.from_string(self.label_template)

            input_texts = [ipt_template.render(input=x) for x in input_text]
            #print(input_texts)

            if self.label_map is None:
                # bypass text labels
                label_texts = labels
            elif isinstance(self.label_map, dict):
                # label-text mapper via dict.
                label_texts = [lbl_template.render(label=self.label_map[x.item()]) for x in labels]
                #print(label_texts)
            elif isinstance(self.label_map, Callable):
                label_texts = [lbl_template.render(label=self.label_map.__call__(x.item())) for x in labels]
            else:
                raise NotImplementedError

            return BatchEncoding(self.tokenizer(text=input_texts, text_target=label_texts,
                                                padding='longest', truncation="only_first",
                                                max_length=self.max_seq_length,
                                                return_tensors="pt", return_attention_mask=True,)
                                 )
        else:
            raise NotImplementedError

        return None
