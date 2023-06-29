#!/usr/bin/env python3
"""
   범용 seq2seq 기반 분류기 평가 코드.
   gold 파일을 읽어서 정답 레이블 셋을 구축하고, test 파일을 읽어서 이를 비교해 gold 내
   모든 레이블에 대해 Acc. 및 F1(Macro/Weighted)를 반환한다.

   copyright (c) 2023~ Jong-hun Shin. ETRI LIRS.
"""

import sys

from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, classification_report

feat_labels = {}

#GOLD_FILE="./mtl_test.answer.top2000.txt"
GOLD_FILE="./mtl_test.answer.txt"
TEST_FILE="./predict_output.full.230629.txt"

golds = []
tests = []

if __name__ == '__main__':
    if len(list(sys.argv)) > 1:
        TEST_FILE=list(sys.argv)[1]

    print(f"** READ GOLD File: {GOLD_FILE}")
    gold_raw = []
    with open(GOLD_FILE, "rt") as gold_f:
        for aline in gold_f:
            aline = aline.strip()
            gold_raw.append(aline)
        gold_f.close()

    # 이제 feature name table을 생성
    kc = Counter(gold_raw)
    gold_label_names = [k for k in kc.keys()]
    gold_label_ids = [i for i in range(len(kc.keys()))]
    feat_labels = dict(zip(gold_label_names, gold_label_ids))

    golds = [feat_labels[item] for item in gold_raw]

    print(f"** READ TEST File: {TEST_FILE}")
    with open(TEST_FILE, "rt") as test_f:
        for aline in test_f:
            aline = aline.strip()
            try:
                tests.append(feat_labels[aline])
            except KeyError:
                tests.append(0)
        test_f.close()

    print(f"golds: #{len(golds)}, tests: #{len(tests)}")

    print(classification_report(golds, tests, labels=gold_label_ids,
                                target_names=gold_label_names, digits=4))
