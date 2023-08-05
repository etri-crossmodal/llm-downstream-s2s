import io
import sys
import zstandard as zstd
import json

from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

import kss
from pororo import Pororo

MAX_INPUT_CTX_LEN=1024
# SPLITS=2면 odd/even으로 masking 함.
# FIXME: SPLITS가 NER 수 보다 많으면 문제가 될 수 있다. 확인해서 수정 필요함..
MASKING_SPLITS=2

DCTX = zstd.ZstdDecompressor(max_window_size=2**31)

def read_lines_from_zst_file(zstd_file_path:Path):
    with zstd.open(zstd_file_path, mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
        for line in iofh:
            yield line

ner = Pororo(task="ner", lang="ko")
file = Path(list(sys.argv)[1])
docs = map(json.loads, read_lines_from_zst_file(file))

output_f = open(f"{list(sys.argv)[1]}-salient_span_masked.jsonl", "wt", encoding="utf-8")

last_sents = ''
for doc in tqdm(docs):
    sents = kss.split_sentences(doc['text1'])
    if last_sents == sents:
        continue
    # 입출력 데이터 쌍을 각각의 string으로 담는다
    input_ctxs, output_lbls = [], []
    extid_nos = []
    current_input_ctx_len = 0
    for i in range(0, MASKING_SPLITS, 1):
        input_ctxs.append([])
        output_lbls.append([])
        extid_nos.append(0)

    for sent in sents:
        blen = len(sent.encode('utf-8'))
        if current_input_ctx_len+blen > MAX_INPUT_CTX_LEN:
            # writedown previous ctxs
            for i in range(0, MASKING_SPLITS, 1):
                if len(input_ctxs[i]) == 0:
                    continue
                output_f.write(json.dumps({'text': ' '.join(input_ctxs[i]), 'label': ' '.join(output_lbls[i])}, ensure_ascii=False) + "\n")
            # 초기화
            input_ctxs, output_lbls = [], []
            extid_nos = []
            current_input_ctx_len = 0
            for i in range(0, MASKING_SPLITS, 1):
                input_ctxs.append([])
                output_lbls.append([])
                extid_nos.append(0)

        orig_sent = deepcopy(sent)
        # truncate sent to 510 chars
        sent = sent[:510]

        inputs = [""] * MASKING_SPLITS
        outputs = [""] * MASKING_SPLITS
        for i in range(0, MASKING_SPLITS, 1):
            inputs[i] = deepcopy(sent)

        ner_list = ([s for s in ner(sent) if s[1] != 'O' ])

        for i in range(0, MASKING_SPLITS, 1):
            for ne in ner_list[i::MASKING_SPLITS]:
                inputs[i] = inputs[i].replace(ne[0], f"<extra_id_{extid_nos[i]}", 1)
                outputs[i] += f" <extra_id_{extid_nos[i]}>" + ne[0]
                extid_nos[i] += 1
            input_ctxs[i].append(inputs[i])
            output_lbls[i].append(outputs[i])

        current_input_ctx_len += blen

    # document 경계를 넘지 않게 flush 해야 함
    if len(input_ctxs[0]) > 0:
        for i in range(0, MASKING_SPLITS, 1):
            if len(input_ctxs[i]) == 0:
                continue
            output_f.write(json.dumps({'text': ' '.join(input_ctxs[i]), 'label': ' '.join(output_lbls[i])}, ensure_ascii=False) + "\n")
        # 초기화
        input_ctxs, output_lbls = [], []
        extid_nos = []
        current_input_ctx_len = 0
        for i in range(0, MASKING_SPLITS, 1):
            input_ctxs.append([])
            output_lbls.append([])
            extid_nos.append(0)

    last_sents = sents

output_f.close()
