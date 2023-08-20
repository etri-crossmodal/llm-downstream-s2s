import string
import re
import os

import tqdm
import evaluate
import numpy as np

from datasets import (load_from_disk, load_dataset, DatasetDict)

from datamodules.klue_datasets import klue_eval_util


if __name__ == '__main__':
    raw_pred_lines = []
    with open("./mrc-pred-test-before-postprocess.tmp.txt", "rt", encoding="utf-8") as in_f:
        # 두 줄이 하나.
        item = ""
        before_insert = True 
        for iidx, aline in enumerate(in_f):
            if aline[:3] == "출력:":
                assert before_insert, f"append again: {iidx}, {aline}"

                item = aline
                before_insert = False
            else:
                assert before_insert is False, f"insert again: {iidx}, {aline}"
                item += aline.strip()
                raw_pred_lines.append(item)
                item = ""
                before_insert = True
        if item != "":
            raw_pred_lines.append(item)
            print(item)
        
                
    # 예측 결과 후처리.
    modified_pred = []
    for pred_out in raw_pred_lines:
        pol = pred_out.split('\n')
        a_pred = pol[0][4:]
        try:
            if pol[1][:3] != '정답임':           # as v2 (23.08.18.01)
                ridx = pol[1].rfind(", 새 정답: ") 
                a_pred = pol[1][ridx+8:]
        except IndexError as e:
            print(pol)
            print(a_pred)
            print(e)
            pass
        modified_pred.append(a_pred)
    INFER_PREDICTIONS = modified_pred

    # 정답 로드 부
    base_kluedata_dir = os.path.abspath(os.path.dirname(__file__))
    base_kluedata_dir += "/datamodules/klue_datasets/"
    mrcds = load_dataset(base_kluedata_dir + "/klue_data.py",
                         name="mrc", data_dir=base_kluedata_dir)
    # for testing purposes.
    #mrcds['test'] = mrcds['test'].shard(num_shards=100, index=99)

    # KLUE-MRC는 collator를 거치지 않으므로, 예측을 이 포맷에 맞게 수정해주면 됨.
    INFER_LABELS = []
    for idx, testdata in enumerate(mrcds['test']):
        newtd = {}
        newtd['id'] = str(idx)
        is_impossible = testdata['plausible_answer']
        ans = testdata['answers']
        if is_impossible:
            #print("is impossible!")
            ans = {'answer_start':[-1], 'text':['[알 수 없음]']}
        else:
            # rename klue mrc answer start_idx to answer_start
            ans = {'answer_start' if k == 'start_idx' else k:v for k, v in ans.items()}
        INFER_LABELS.append({'id': str(idx), 'answers': ans})
    INFER_PREDICTIONS = [{'prediction_text': apred, 'id': str(idx)}
                                     for idx, apred in enumerate(INFER_PREDICTIONS)]

    # KLUE 평가 방법을 사용한 평가: EM/ROUGE-W
    em_scores, rouge_scores = [], []
    for idx, v in enumerate(INFER_PREDICTIONS):
        pred_answer = v['prediction_text']
        pred_answer = klue_eval_util.normalize_answer_for_klue_mrc(pred_answer)
        ground_truths = [klue_eval_util.normalize_answer_for_klue_mrc(atruth)
                         for atruth in INFER_LABELS[idx]['answers']['text']]
        #print(f"pred_answer: {pred_answer}")
        #print(f"ground truths: {str(ground_truths)}")

        em, rouge = klue_eval_util.compute_em_and_rouge_w_score_for_klue_mrc(pred_answer, ground_truths)
        em_scores.append(em)
        rouge_scores.append(rouge)

    print(f'(Official) KLUE MRC Eval - "exact_match": {np.mean(em_scores)}, "rouge": {np.mean(rouge_scores)}')

    # hf evaluate를 사용한 평가.
    squad_metric = evaluate.load("squad")
    squad_res = squad_metric.compute(references=INFER_LABELS,
                                     predictions=INFER_PREDICTIONS)
    print(f"SQuAD Metrics - {str(squad_res)}")

    # chrf, rouge는 references=[[str],], predictions=[str,] 을 받는다
    refs = [lbl['answers']['text'] for lbl in INFER_LABELS]
    preds = [prd['prediction_text'] for prd in INFER_PREDICTIONS]

    """
    # chrF는 backend인 sacrebleu의 요구사항대로, references 갯수가 모두 같아야 한다.
    # 그래서 제외됨.
    chrf_metric = evaluate.load("chrf")
    chrf_res = chrf_metric.compute(references=refs, predictions=preds)
    print(f"chrF - {str(chrf_res)}")
    """

    rouge_metric = evaluate.load("rouge")
    rouge_res = rouge_metric.compute(references=refs, predictions=preds)
    print(f"ROUGE Metrics - {str(rouge_res)}")
