import re
import os
import json

from tqdm import tqdm

def korquadv1_convert(input_f, out_f):
    jd = json.load(input_f)
    # depth 1 - jd['version'], jd['data'] 로 구성
    jdd = jd['data']
    for jd_elem in jdd:
        # depth 2 - 'paragraphs', 'title' 로 구성
        for paragraph in tqdm(jd_elem['paragraphs'], total=len(jd_elem['paragraphs'])):
            # depth 3 - qas, context
            passage = paragraph['context']
            # passage split
            splitted_passage = re.sub('([.?!]) ', '\\1**CUT**', passage).split('**CUT**')
            for qa in paragraph['qas']:
                # depth 4 - answers, id, question
                idstr = qa['id']
                ques_str = qa['question']
                # answer는 position, label에 각각 모은다
                labels = []
                positions = []
                for answer in qa['answers']:
                    answer_str = answer['text']
                    labels.append(answer_str)
                    answer_pos = answer['answer_start']
                    answer_sent = ""
                    # position 정보는 (시작위치, '문장')의 tuple로 구성
                    current_pos = 0
                    for a_sent in splitted_passage:
                        if answer_pos > current_pos and answer_pos < current_pos+len(a_sent):
                            answer_sent = a_sent
                        current_pos += len(a_sent)
                    positions.append((answer_pos, answer_sent))
                # 저장 위치
                doc = {'id': idstr, 'type':'mrc', 'text1': passage, 'text2': ques_str,
                       'position': positions, 'label': labels}
                #print(doc)
                out_f.write(json.dumps(doc, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # 출처 KorQuAD 타입 MRC 문제 문어 이름 v1.0 기계독해 언어 KR 상용 N
    out_f = open("korquad_v1.0-train.jsonl", "wt")
    with open("KorQuAD_v1.0_train.json", "rt") as input_f:
        korquadv1_convert(input_f, out_f)

    out_f.close()

    out_f = open("korquad_v1.0-valid.jsonl", "wt")
    with open("KorQuAD_v1.0_dev.json", "rt") as input_f:
        korquadv1_convert(input_f, out_f)

    out_f.close()
