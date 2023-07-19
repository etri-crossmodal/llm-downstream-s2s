"""
    Data Collator of a Enc-Dec Structure NNs, for KorQUAD (v1) dataset.

    Copyright (C) 2023~, Jong-hun Shin. ETRI LIRS.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Union, Optional, Dict

from transformers import AutoTokenizer, BatchEncoding


@dataclass
class KorQuadV1DataCollator:
    """
        KorQuadV1 data collator.
    """
    # dataclass definitions
    tokenizer: Optional[Callable]=field(
        default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=None
    max_seq_length: Optional[int]=None

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(examples, dict):
            contexts = examples['context']
            questions = examples['question']
            labels = examples['label']

            input_texts = []
            label_texts = []
            for idx, ques in enumerate(questions):
                # 가장 간단한 접근 - 길이가 문제다. 학습 때는 이렇게 해서는 안됨.
                input_texts.append(f"task: MRC\n\nquestion: {ques}\n\n"
                                   f"context: {contexts[idx]}\n")
                # set shortest label data
            for lbl in labels:
                label_texts.append(lbl[0])

            return BatchEncoding(self.tokenizer(text=input_texts, text_target=label_texts,
                                                padding='longest', truncation="only_first",
                                                max_length=self.max_seq_length,
                                                return_tensors="pt", return_attention_mask=True,)
                                 )
        else:
            raise NotImplementedError

        return None
