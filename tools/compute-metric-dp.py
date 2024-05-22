"""
    Dependency Parsing의 평가를 위한 스크립트.

    한 입력에 대한 정답은 2줄로, 첫번째 줄은 lemmatize 출력(형태소 분석 결과) + 어휘 인덱스 + 품사
    두번째 줄은 lemmatize 출력 + 헤드 번호 + 의존 레이블 로 구성되어 있다.

    구조분석 결과의 시작줄은 "deprel: "로 시작한다.

    이해-생성 모델이므로, 입력의 lemma/word_form 수와 다르게 나타날 수 있고,
    파싱도 실패할 수 있다. 이 때문에, 정답의 길이보다 긴 예측은 정답의 길이만큼 잘라냈고,
    정답의 길이보다 짧은 예측은 -2(BAD)로 채워넣는다.

    LAS는 gold의 head != pred의 head면 pred의 deprel을 -2(=BAD)로 바꾼 뒤,
    sklearn.metric.f1_score를 사용하여 macro average F1을 KLUE 공식 수치로 사용한다.

    LAS 계산을 위해서, deprel은 인덱스 번호가 15보다 크거나 같으면 -3으로 통일하여
    계산을 수행한다. (as https://arxiv.org/pdf/2105.09680.pdf, page 35.)
    see also - https://github.com/KLUE-benchmark/KLUE-baseline/blob/main/klue_baseline/metrics/functional.py:141

    Copyright (C) 2023~ Jong-hun Shin. ETRI LIRS.
"""
import re
import numpy as np

from sklearn.metrics import f1_score

# v2
#GOLD_FILENAME = "./klue_dp_gold_out-with-input_lemma.txt"
#PRED_FILENAME = "./klue_dp_pred_out_gbst-base-with-input_lemma-230814.txt"

# v3
#GOLD_FILENAME = "./klue_dp_gold_out-v3.txt"
#PRED_FILENAME = "./klue_dp_pred_out_gbst-base-v3-230815.txt"

# v3a
#GOLD_FILENAME = "./klue_dp_gold_out-v3a.txt"
#PRED_FILENAME = "./klue_dp_pred_out_gbst-base-v3a-230815.txt"

# v3b
#GOLD_FILENAME = "./klue_dp_gold_out-v3b.txt"
#PRED_FILENAME = "./klue_dp_pred_out_gbst-base-v3b-230815.txt"

# v3c, v3d
GOLD_FILENAME = "./klue_dp_gold_out-v3d.txt"
PRED_FILENAME = "./klue_dp_pred_out_gbst-base-1144k-ssm-lr7e-5-v3d-240521.txt"

# 구조분석 레이블. KLUE-baseline 코드에서 가져옴.
# https://github.com/KLUE-benchmark/KLUE-baseline/blob/8a03c9447e4c225e806877a84242aea11258c790/klue_baseline/data/klue_dp.py#L490
dep_labels = ["NP", "NP_AJT", "VP", "NP_SBJ", "VP_MOD",
              "NP_OBJ", "AP", "NP_CNJ", "NP_MOD", "VNP",
              "DP", "VP_AJT", "VNP_MOD", "NP_CMP", "VP_SBJ",
              "VP_CMP", "VP_OBJ", "VNP_CMP", "AP_MOD", "X_AJT",
              "VP_CNJ", "VNP_AJT", "IP", "X", "X_SBJ",
              "VNP_OBJ", "VNP_SBJ", "X_OBJ", "AP_AJT", "L",
              "X_MOD", "X_CNJ", "VNP_CNJ", "X_CMP", "AP_CMP",
              "AP_SBJ", "R", "NP_SVJ",
             ]


def read_predicts(filename):
    lemmas = []                # lemma는 아직 사용하지 않으니 파싱되지 않은 줄만 삽입
    dep_heads = []             # [[헤드, ..., ...], [헤드, ..., ...], ...]
    dep_deprels = []           # [[의존레이블, ..., ...], [의존레이블, ..., ,...], ...]
    parse_failure = 0

    lines = 0
    with open(filename, "rt", encoding="utf-8") as in_f:
        for aline in in_f:
            lines += 1
            if aline[:5] == "lemma":
                # lemma line
                lemmas.append(aline.strip()[6:])
            elif aline[:5] == 'depre':
                # dep line
                aline = aline[8:]
                #print(aline)
                deps = aline.split('▁')
                heads_list = []
                deps_list = []
                for dep in deps:
                    mat = re.match("\((.+?), ([0-9]+), ([A-Z_]+)\)", dep)
                    if mat is None:
                        print(f"cannot parsable. skip this prediction. {dep}")
                        parse_failure += 1
                    #    continue
                    assert mat is None or len(mat.groups()) == 3, f"matched groups not 3. parse failure. {dep}"
                    heads_list.append(-2 if mat is None else int(mat.groups()[1]))
                    if mat is not None:
                        lbl = mat.groups()[2]
                        if lbl in dep_labels:
                            lbl_value = dep_labels.index(lbl)
                        else:
                            lbl_value = -2          # 레이블이 없으면 -2 = BAD로

                        if lbl_value >= 15:
                            lbl_value = -3          # 15가 넘어가면 -3으로 통일
                                                    # same as https://github.com/KLUE-benchmark/
                                                    #                 KLUE-baseline/blob/main/
                                                    #                 klue_baseline/metrics/functional.py:141
                    deps_list.append(lbl_value)
                dep_heads.append(heads_list)
                dep_deprels.append(deps_list)
            else:
                # for v3a --> word_counts: int(x)\n
                continue

    #print(f"{lines}")

    print(f"Parse failures: #{parse_failure} errors occured.")
    return lemmas, dep_heads, dep_deprels


if __name__ == '__main__':
    # 순서는 pred, gold 순서
    if list(sys.argv) > 2:
        PRED_FILENAME = list(sys.argv)[1]
        GOLD_FILENAME = list(sys.argv)[2]

    # 정답파일 load
    print(f"read gold: {GOLD_FILENAME}")
    gold_lemmas, gold_heads, gold_deprels = read_predicts(GOLD_FILENAME)
    print(f"read pred: {PRED_FILENAME}")
    pred_lemmas, pred_heads, pred_deprels = read_predicts(PRED_FILENAME)

    assert len(gold_heads) == len(pred_heads), f"gold heads: {len(gold_heads)}, pred_heads: {len(pred_heads)}"

    aligned_pred_heads, aligned_pred_deprels = [], []

    shorter_pred_head, longer_pred_head = 0, 0
    shorter_pred_deprel, longer_pred_deprel = 0, 0

    # 개수를 맞춰준다. 헤드가 없으면 -2을 채우고
    for idx, gold_head in enumerate(gold_heads):
        gold_deprel = gold_deprels[idx]
        pred_head = pred_heads[idx]
        pred_deprel = pred_deprels[idx]

        # 짧으면 채우고 길면 잘라낸다.
        if len(gold_head) > len(pred_head):
            print(f'shorter case, gold({idx}) head: {gold_head}')
            print(f'shorter case, pred({idx}) head: {pred_head}')
            pred_head += [-2] * (len(gold_head)-len(pred_head))
            shorter_pred_head += 1
        elif len(gold_head) < len(pred_head):
            pred_head = pred_head[:len(gold_head)]
            longer_pred_head += 1

        if len(gold_deprel) > len(pred_deprel):
            print(f'shorter case, gold({idx}) deprel: {gold_deprel}')
            print(f'shorter case, pred({idx}) deprel: {pred_deprel}')
            pred_deprel += [-2] * (len(gold_deprel)-len(pred_deprel))
            shorter_pred_deprel += 1
        elif len(gold_deprel) < len(pred_deprel):
            print(f'longer case, gold({idx}) deprel: {gold_deprel}')
            print(f'longer case, pred({idx}) deprel: {pred_deprel}')
            pred_deprel = pred_deprel[:len(gold_deprel)]
            longer_pred_deprel += 1

        assert len(pred_head) == len(gold_head)
        assert len(pred_deprel) == len(gold_deprel)

        # gold-head, pred-head 가 틀린 곳은 pred-deprel도 다른 타입으로 바꿔줘야 함
        for ii, gh in enumerate(gold_head):
            if gh != pred_head[ii]:
                pred_deprel[ii] = -2

        aligned_pred_heads.append(pred_head)
        aligned_pred_deprels.append(pred_deprel)

    print(f"shorter pred head: {shorter_pred_head}, longer pred head: {longer_pred_head}")
    print(f"shorter pred deprel: {shorter_pred_deprel}, longer pred deprel: {longer_pred_deprel}")

    # 이제 flatten
    gold_head_flatten = [item for sublist in gold_heads for item in sublist]
    pred_head_flatten = [item for sublist in aligned_pred_heads for item in sublist]

    gold_deprel_flatten = [item for sublist in gold_deprels for item in sublist]
    pred_deprel_flatten = [item for sublist in aligned_pred_deprels for item in sublist]


    print(f"highest UAS macro F1: KLUE-RoBERTa-Large: 93.84, LAS macro F1: 87.93")
    print(f"UAS macro F1 (official): {f1_score(gold_head_flatten, pred_head_flatten, average='macro') * 100.0}")
    print(f"UAS micro F1: {f1_score(gold_head_flatten, pred_head_flatten, average='micro') * 100.0}")

    print(f"LAS macro F1 (official): {f1_score(gold_deprel_flatten, pred_deprel_flatten, average='macro') * 100.0}")
    print(f"LAS micro F1: {f1_score(gold_deprel_flatten, pred_deprel_flatten, average='micro') * 100.0}")
