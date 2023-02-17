"""
    Data Collator of a Enc-Dec Structure NNs, for Test Data.

    Copyright (C) 2023~, Jong-hun Shin. ETRI LIRS.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Union, Optional, Dict

from transformers import AutoTokenizer, ByT5Tokenizer, BatchEncoding

@dataclass
class korailCollatorV1:
    # dataclass definitions
    sent1_field_name: str="title"
    sent2_field_name: str="content"
    label_field_name: str="label"
    tokenizer: Optional[Callable]=field(default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    # 0: 같지 않음, 1: 같음
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=field(default_factory=lambda: {0:'different', 1:'paraphrase'})
    length_limit: int=2048

    def __call__(self, examples: dict[str, Any]) -> dict[str, Any]:
        if isinstance(examples, dict):
            sent1s = examples[self.sent1_field_name]
            sent2s = examples[self.sent2_field_name]
            labels = examples[self.label_field_name]

            input_texts = []
            for idx, sent1 in enumerate(sent1s):
                s2 = sent2s[idx]
                input_text = f"#TITLE {sent1}\n#CONTENT {s2}"
                input_text = input_text.encode('utf-8')[:self.length_limit-1].decode('utf-8', 'ignore')
                input_texts.append(input_text)

            if isinstance(self.label_map, dict):
                # label-text mapper via dict.
                label_texts = [self.label_map[x.item()] for x in labels]
            elif isinstance(self.label_map, Callable):
                label_texts = [self.label_map(x.item()) for x in labels]
            else:
                raise NotImplemented

            #print(f"{input_texts[0]}\n=> {label_texts[0]}")
            return BatchEncoding(self.tokenizer(text=input_texts, text_target=label_texts, padding='longest', return_tensors="pt"))
        else:
            raise NotImplemented

        return None
