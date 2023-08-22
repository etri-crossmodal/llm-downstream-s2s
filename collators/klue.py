"""
    Data Collator of a Enc-Dec Structure NNs, for KLUE datasets.

    Copyright (C) 2023~, Jong-hun Shin. ETRI LIRS.
"""
import re
import random

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
    tokenizer: Optional[Callable]=field(
        default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    # 0: 함의/entailment, 1: 중립/neutral, 2: 모순/contradiction
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=field(
        default_factory=lambda: {0:'함의', 1:'중립', 2:'모순'})
    max_seq_length: Optional[int]=None

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
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
                #input_texts.append(f"다음 전제-가설간의 관계가 함의면 'entailment', "
                #                   f"중립이면 'neutral', 모순이면 'contradiction'를 출력해라:\n\n"
                #                   f"전제: {premise}\n가설: {hyp}")
                input_texts.append(f"NLI, 전제: {premise}\n가설: {hyp}\n")

            if isinstance(self.label_map, dict):
                # label-text mapper via dict.
                label_texts = [ self.label_map[x.item()] for x in labels]
            elif isinstance(self.label_map, Callable):
                label_texts = [ self.label_map(x.item()) for x in labels]
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


@dataclass
class KLUEYNATDataCollator:
    """
        KLUE-YNAT Data Collator.
        테스트용. 원래는 기사도 가져와서 결정해야 함
    """
    # dataclass definitions
    title_field_name: str="title"
    label_field_name: str="label"
    tokenizer: Optional[Callable]=field(
        default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=field(
        default_factory=lambda: {0:'IT과학', 1:'경제',
                                 2:'사회', 3:'생활문화', 4:'세계',
                                 5:'스포츠', 6:'정치'})
    max_seq_length: Optional[int]=None

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
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

            return BatchEncoding(self.tokenizer(text=input_texts, text_target=label_texts,
                                                padding='longest', truncation="only_first",
                                                max_length=self.max_seq_length,
                                                return_tensors="pt", return_attention_mask=True,)
                                 )
        else:
            raise NotImplementedError

        return None


@dataclass
class KLUEMRCDataCollator:
    """
        KLUE-MRC data collator.
    """
    # dataclass definitions
    tokenizer: Optional[Callable]=field(
        default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=None
    max_seq_length: Optional[int]=None

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(examples, dict):
            titles = examples['title']
            contexts = examples['context']
            questions = examples['question']
            labels = examples['answers']
            is_impossibles = examples['plausible_answer']

            input_texts = []
            label_texts = []
            for idx, title in enumerate(titles):
                ctx = contexts[idx]
                is_impos = is_impossibles[idx]
                # 지문을 문장단위로 쪼개어 준다. loose하게 쪼개자. 복수개의 문장이 잡혀도 그게 나음.
                ctx_lists = [f"{ix} // {x}" for ix, x in enumerate(
                    [x.strip() for x in re.split("(.+?[가-힣]{3}[\.?!]+)( +|$)", ctx) if x != '' and len(x) > 1])]
                #ctx_lists = [f"{ix} // {x}" for ix, x in enumerate(kss.split_sentences(ctx,))]
                #ctx = '\n'.join(ctx_lists)

                # 가장 간단한 접근 - 길이가 문제다. 학습 때는 이렇게 해서는 안됨.
                input_texts.append(f"task: MRC, 지문을 읽고 질문에 답을 할 수 없다면 '[알 수 없음]'을 출력한다.\n"
                                   f"질문: {questions[idx]}\n"
                                   f"지문: {ctx}\n")
                # 비슷한 질문을 찾아보고 답이 있나 없나를 보게 할 것인가?
                # 아니면 그냥 관련있는 지문이 없다고 하고 끝 낼 것인가?
                """
                lbl_included_ctx_idx = []
                if is_impos:
                    label_base = "-1 // 정답 없음"
                else:
                    for ixx, ct in enumerate(ctx_lists):
                        if ct.replace(' ', '').find(labels['text'][idx].replace(' ', '')) >= 0:
                            lbl_included_ctx_idx.append(str(ixx))
                    if len(lbl_included_ctx_idx) > 0:
                        label_base = ','.join(lbl_included_ctx_idx) + f" // {labels['text'][idx]}"
                    else:
                        # 그 수가 많지 않으니 강제로 정답이 없다고 할 것.
                        label_base = f"-1 // 정답 없음}"
                        print("** WARNING: klue-mrc, label text not found in contexts, but not impossibles.")
                        print(f"{ctx}")
                        print(f"정답: {labels['text'][idx]}") 
                """
                if is_impos:
                    # 정답이 나올 수 없으면 다음과 같이 한다:
                    # (1) 오답 + 정답이 아님
                    # (1) '알 수 없음' + 정답임
                    if random.randrange(0, 100) > 20:
                        label_base = "출력: [알 수 없음]\n정답임"
                    else:
                        # 이렇게 숙의를 하게 하는게 맞는건가? 아니면 정답을 냈을 때 아니라고 하고 덮어 쓰는게 맞는가?
                        label_base = f"출력: {labels['text'][idx]}\n정답이 아님, 새 정답: [알 수 없음]" 
                else:
                    # 이 경우에도 가끔씩 '알 수 없음' + 정답이 아님 을 출력하게 해야 함. 3%를 하게 하자.
                    # 숫자는 전체의 1/10에 적용
                    choice_val = random.randrange(0, 100)
                    if choice_val < 3 or \
                            (re.search("[0-9]", labels['text'][idx]) is not None and random.randrange(0, 10) < 1):
                        albl = labels['text'][idx]
                        # add random noise - replace character with some constraints 
                        # 일단 바꿀 위치를 결정하고 replace만 해 본다. delete/insertion도 해 봐야 하나?
                        repl_pos = random.randrange(0, len(albl))
                        if re.match("[0-9]", albl[repl_pos]) is not None:
                            cand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                            del cand[int(re.match("[0-9]", albl[repl_pos]).group()[0])]
                            albl = albl[:repl_pos] + str(random.choice(cand)) + albl[repl_pos+1:]
                        else:
                            albl = albl[:repl_pos] + chr(random.randrange(0xac00, 0xd7af)) + albl[repl_pos+1:]
                        # [0-9]이면 다른 숫자로 결정
                        # then add
                        label_base = f"출력: {albl}\n정답이 아님, 새 정답: {labels['text'][idx]}"
                    elif choice_val < 23:
                        #label_base = f"출력: [알 수 없음]\n정답이 아님" # 이러면 정답을 다시 내놓을 수 가 없게 된다.
                        # 아래 주석을 풀고 추가 실험할 것.
                        label_base = f"출력: [알 수 없음]\n정답이 아님, 새 정답: {labels['text'][idx]}"
                    else:
                        label_base = f"출력: {labels['text'][idx]}\n정답임"
                label_texts.append(label_base)

            return BatchEncoding(self.tokenizer(text=input_texts, text_target=label_texts,
                                                padding='longest', truncation="only_first",
                                                max_length=self.max_seq_length,
                                                return_tensors="pt", return_attention_mask=True,)
                                 )
        else:
            raise NotImplementedError

        return None


@dataclass
class KLUENERDataCollator:
    """
        KLUE-NER data collator.
    """
    # dataclass definitions
    tokenizer: Optional[Callable]=field(
        default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=None
    max_seq_length: Optional[int]=None

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(examples, dict):
            sentences = examples['tagged_sent']

            input_texts = []
            label_texts = []
            for idx, sentence in enumerate(sentences):
                notag_sent = re.sub("<(.+?):[A-Z][A-Z]>", "\\1", sentence)
                only_tags = ' '.join(re.findall("<.+?:[A-Z][A-Z]>", sentence))
                input_texts.append(f"task: NER\n\nInput: {notag_sent}\n")
                # set shortest label data
                label_texts.append(sentence)
                #label_texts.append(only_tags)

            assert len(input_texts) == len(label_texts)

            return BatchEncoding(self.tokenizer(text=input_texts, text_target=label_texts,
                                                padding='longest', truncation="only_first",
                                                max_length=self.max_seq_length,
                                                return_tensors="pt", return_attention_mask=True,)
                                 )
        else:
            raise NotImplementedError

        return None


@dataclass
class KLUEDPDataCollator:
    """
        KLUE-DP data collator.
    """
    # dataclass definitions
    tokenizer: Optional[Callable]=field(
        default_factory=lambda: AutoTokenizer.from_pretrained("google/byt5-small", return_tensors="pt"))
    label_map: Union[Dict[Any, str], Callable[Any,Any]]=None
    max_seq_length: Optional[int]=None

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(examples, dict):
            sentences = examples['sentence']
            ma_outs = examples['ma_out']
            labels = examples['label']

            input_texts = []
            label_texts = []
            for idx, sentence in enumerate(sentences):
                input_texts.append(f"task: Dependency Parsing\n\nInput: {sentence}\nLemmas: {ma_outs[idx]}")
                label_texts.append(labels[idx])

            assert len(input_texts) == len(label_texts)

            return BatchEncoding(self.tokenizer(text=input_texts, text_target=label_texts,
                                                padding='longest', truncation="only_first",
                                                max_length=self.max_seq_length,
                                                return_tensors="pt", return_attention_mask=True,)
                                 )
        else:
            raise NotImplementedError

        return None
