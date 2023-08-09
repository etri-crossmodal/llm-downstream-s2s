"""
    레이블만 주어진 상태에서 F1^e(entity-level F1) 를 계산하는 코드.
    Char F1(F1^c)은 전체 문장이 포함되어 있어야 올바르게 계산될 것이다.

    Copyright (C) 2023~ Jong-hun Shin. ETRI LIRS.
"""
import sys
import re

import evaluate

if len(list(sys.argv)) < 2:
    print(f"Usage: {list(sys.argv)[0]} [predict_ner_label.txt] [gold_ner_label.txt]")
    sys.exit(-1)

# load predicted one
preds = []
pred_sents = []
pred_iob2s = []     # 'O' tag는 포함하지 않고 레이블만 본다.
with open(list(sys.argv)[1], 'rt', encoding='utf-8') as in_f:
    is_whole_sent = False
    for aline in in_f:
        aline = aline.strip()
        pred_sents.append(aline)
        # 만약 predicted one이 whole sentence일 경우 label만 뽑아낸다, 그 뒤 F1^e를 계산
        only_tags = '▁'.join(re.findall("<.+?:[A-Z][A-Z]>", aline))
        alist = only_tags.split('▁')
        apred = []
        apred_iob2 = []
        for elem in alist:
            if len(elem) == 0:
                continue
            if elem[-4] != ':':
                print(f"in parse preds, not tag found: {elem}, skip this label.")
                continue
            wrd = elem[1:-4]
            tag = elem[-3:-1]
            for idx, char in enumerate(wrd):
                prefix = "B" if idx == 0 else "I"
                apred_iob2.append(prefix + '-' + tag.upper())
            apred.append((wrd, tag))
        # 빈 엘리먼트를 하나 넣어준다
        if len(apred) == 0:
            apred = [('','')]
        preds.append(apred)
        pred_iob2s.append(' '.join(apred_iob2))

golds = []
gold_sents = []
gold_iob2s = []
with open(list(sys.argv)[2], 'rt', encoding='utf-8') as in_f:
    for aline in in_f:
        aline = aline.strip()
        gold_sents.append(aline)
        only_tags = '▁'.join(re.findall("<.+?:[A-Z][A-Z]>", aline))
        alist = only_tags.split('▁')
        agold = []
        agold_iob2 = []
        for elem in alist:
            if elem[-4] != ':':
                print(f"in parse golds, not tag found: {elem}, skip this label")
                continue
            wrd = elem[1:-4]
            tag = elem[-3:-1]
            for idx, char in enumerate(wrd):
                prefix = "B" if idx == 0 else "I"
                agold_iob2.append(prefix + '-' + tag.upper())
            agold.append((wrd, tag))
        golds.append(agold)
        gold_iob2s.append(' '.join(agold_iob2))


assert len(preds) == len(golds), f"preds({len(preds)}) != golds({len(golds)})"

def _compute_f1_metrics(whole_preds, whole_golds):
    """ 예측된 텍스트:NER클래스 레이블을 entity 수준에서 F1을 계산한다.
        https://github.com/paust-team/pko-t5/blob/main/pkot5/klue/processors.py:224 에서
        계산 코드를 가져옴
    """
    num_correct, num_golds, num_preds = 0, 0, 0

    for idx, pred in enumerate(whole_preds):
        num_golds += len(whole_golds[idx])
        num_preds += len(pred)

        preds = {text: label for text, label in pred}
        for text, label in whole_golds[idx]:
            if text in preds:
                pred_label = preds[text]
                if label == pred_label:
                    num_correct += 1

    prec = num_correct / num_preds
    recall = num_correct / num_golds
    f1 = (2 * prec * recall) / (prec + recall + 1e-8)
    return { "NER, f1^e(Entities F1)": f1 }

print(_compute_f1_metrics(preds, golds))

# char-F1 은 언어 이해 모델에서는 입력-출력 시퀀스 길이가 항상 같기 때문에 문제가 되지 않지만,
# 생성 모델이나 이해생성모델에서는 시퀀스 길이가 상이하다. 원래 KLUE-NER의 CharF1은 부분매칭 성능을 보기 위함이다.
# (모델이 얼마나 stem과 affixes를 잘 분리하는지를 보기위해.)
# 원문:  Char F1 score is newly provided to measure a partial overlap between a model prediction and a ground truth. We additionally report this measure to see how well a model decomposes stems and affixes in Korean, which significantly affects the model performance of NER.  Char F1 is an average of class-wise F1-scores.
# Class ["B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG", "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT", "O"]
# 여기서, majority negative entity class (O)를 제거하여 평가한다고 했다.
# 그래서, 레이블만을 IOB2 시퀀스로 바꾼 다음에, chrf(Popovic et al., '15)에 word-level f1으로 계산함.
# beta term을 1로 두어, 보통의 word-level unigram F1으로 계산함
chrf_eval = evaluate.load("chrf")
print(f"IOB2-only word-F1: {chrf_eval.compute(references=gold_iob2s, predictions=pred_iob2s, char_order=0, word_order=1, beta=1)}")

# 레이블 수준의 chrf를 구하면 다음과 같다:
# gold: <신:PS> <신동빈:PS> <신:PS> <롯데그룹:OG>
# VS.
# pred: <신:PS> <신동빈:PS> <신:PS> <롯데:OG>
# 이것을 chrF1 (6-gram character F1, beta=1) 으로 계산함.
print(f"chrF: {chrf_eval.compute(references=gold_sents, predictions=pred_sents, beta=1)}")
