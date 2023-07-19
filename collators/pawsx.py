"""
    Data Collator of a Enc-Dec Structure NNs, for PaWS-X (arXiv:1908.11828)

    Copyright (C) 2023~, Jong-hun Shin. ETRI LIRS.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Union, Optional, Dict

from transformers import AutoTokenizer, ByT5Tokenizer, BatchEncoding

@dataclass
class PAWS_XDataCollator:
    # dataclass definitions
    sent1_field_name: str="sentence1"
    sent2_field_name: str="sentence2"
    label_field_name: str="label"
    tokenizer: Optional[Callable]=field(
        default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    # 0: 같지 않음, 1: 같음
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=field(
        default_factory=lambda: {0:'different', 1:'paraphrase'})
    max_seq_length: Optional[int]=None

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(examples, dict):
            sent1s = examples[self.sent1_field_name]
            sent2s = examples[self.sent2_field_name]
            labels = examples[self.label_field_name]

            input_texts = []
            for idx, sent1 in enumerate(sent1s):
                s2 = sent2s[idx]
                input_texts.append(f"paraphrase detection task:\n\nsentence-1: {sent1}\nsentence-2: {s2}\n")

            if isinstance(self.label_map, dict):
                # label-text mapper via dict.
                label_texts = [self.label_map[x.item()] for x in labels]
            elif isinstance(self.label_map, Callable):
                label_texts = [self.label_map(x.item()) for x in labels]
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
