"""
    예측된 레이블을 원래대로 NER tags가 포함된 원문으로 복원 시도하는 코드.
    잘 안되었음... 그냥 바로 텍스트-레이블 쌍을 찾게 하는게 맞겠다.
"""
import re
import copy

orig_sents = []
with open('./klue-ner-devset-naive-input-prompted.txt', 'rt') as in_f:
    for aline in in_f:
        aline = aline.strip()
        orig_sents.append(aline)

preds = []
with open('./gbst-base-ssm-stage2-ner-labelonly-predict-230808.txt', 'rt') as in_f:
    for aline in in_f:
        aline = aline.strip()
        aline = aline.replace('> <', '>▁<')
        alist = aline.split('▁')
        preds.append(alist)

golds = []
with open('./klue-ner-devset-gold-230808-wholesent.txt', 'rt') as in_f:
    for aline in in_f:
        aline = aline.strip()
        golds.append(aline)

sents = copy.deepcopy(orig_sents)


# 아직 문제가 많다... ㅠㅜ
for i in range(len(sents)):
    start_pos = 0
    for elem in preds[i]:
        wrd = elem[1:-4]
        tag = elem[-3:-1]
        # 찾는 부분에서 겹치는 경우가 있다. 이걸 막기 위해서는 정규식 '^wrd' 또는 ' wrd' 매치가 필요하다.
        pos = sents[i].find(wrd, start_pos)

        # 그래서 수동으로 보정

        if sents[i].count(wrd, start_pos) > 1 and pos > 0 and \
                sents[i][pos-1] not in [' ', "'", '"', '(', '[', ',', '.', ')', '·', ' ', '+',
                                        '비', '즈',
                                        '면', '별', '리', '말', ']', '>', '<', '&', '-', '→', ]:
            pos = sents[i].find(wrd, start_pos+pos)
            # 몇 번 더
            if sents[i][pos-1] not in [' ', "'", '"', '(', '[', ',', '.', ')', '·', ' ', '+',
                                       '비', '즈',
                                       '면', '별', '리', '말', ']', '>', '<', '&', '-', '→', ]:
                pos = sents[i].find(wrd, start_pos+pos)

        if pos != -1:
            sents[i] = sents[i][:pos] + f'<{wrd}:{tag}>' + sents[i][pos+len(wrd):]
            start_pos = pos+len(wrd)+5

differs = 0
for i in range(len(sents)):
    if sents[i] != golds[i]:
#        print(f"differ found: {i}")
#        print(preds[i])
#        print(sents[i])
#        print(golds[i])
        differs += 1

print(f"total differs: {differs}/{len(sents)}")
