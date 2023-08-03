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


DCTX = zstd.ZstdDecompressor(max_window_size=2**31)

def read_lines_from_zst_file(zstd_file_path:Path):
    with zstd.open(zstd_file_path, mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
        for line in iofh:
            yield line

ner = Pororo(task="ner", lang="ko")
file = Path(list(sys.argv)[1])
docs = map(json.loads, read_lines_from_zst_file(file))
MAX_INPUT_CTX_LEN=1024

output_f = open(f"{list(sys.argv)[1]}-salient_span_masked.jsonl", "wt", encoding="utf-8")

last_sents = ''
for doc in tqdm(docs):
    sents = kss.split_sentences(doc['text1'])
    if last_sents == sents:
        continue
    # 입출력 데이터 쌍을 각각의 string으로 담는다
    odd_input_ctx, odd_output_lbl, even_input_ctx, even_output_lbl = [], [], [], []
    current_input_ctx_len, odd_extid_no, even_extid_no = 0, 0, 0

    for sent in sents:
        blen = len(sent.encode('utf-8'))
        if current_input_ctx_len+blen > MAX_INPUT_CTX_LEN:
            # writedown previous ctxs
            output_f.write(json.dumps({'text': ' '.join(odd_input_ctx), 'label': ' '.join(odd_output_lbl)}, ensure_ascii=False) + "\n")
            output_f.write(json.dumps({'text': ' '.join(even_input_ctx), 'label': ' '.join(even_output_lbl)}, ensure_ascii=False) + "\n")
            odd_input_ctx, odd_output_lbl, even_input_ctx, even_output_lbl = [], [], [], []
            current_input_ctx_len, odd_extid_no, even_extid_no = 0, 0, 0

        orig_sent = deepcopy(sent)
        # truncate sent to 510 chars
        sent = sent[:510]
        odd_sent = deepcopy(sent)
        even_sent = deepcopy(sent)

        odd_output = ""
        even_output = ""

        ner_list = ([s for s in ner(sent) if s[1] != 'O' ])

        # odd
        for ne in ner_list[1::2]:
            # 마스킹 강도를 조절할 필요가 있다: 예) 위치를 먼저 찾고, 이전에 >가 있으면 skip, 아니면 1회만 나타나도록 combinatorial 하게 하던지..
            # 아니면 한 sentence만 하도록 하던지?
            odd_sent = odd_sent.replace(ne[0], f"<extra_id_{odd_extid_no}>", 1)
            odd_output += f" <extra_id_{odd_extid_no}>" + ne[0]
            odd_extid_no += 1
        odd_input_ctx.append(odd_sent)
        odd_output_lbl.append(odd_output)

        # even
        for ne in ner_list[0::2]:
            # 마스킹 강도를 조절할 필요가 있다: 예) 위치를 먼저 찾고, 이전에 >가 있으면 skip, 아니면 1회만 나타나도록 combinatorial 하게 하던지..
            # 아니면 한 sentence만 하도록 하던지?
            even_sent = even_sent.replace(ne[0], f"<extra_id_{even_extid_no}>", 1)
            even_output += f" <extra_id_{even_extid_no}>" + ne[0]
            even_extid_no += 1
        even_input_ctx.append(even_sent)
        even_output_lbl.append(even_output)

        current_input_ctx_len += blen

    # document 경계를 넘지 않게 flush 해야 함
    if len(odd_input_ctx) > 0:
        output_f.write(json.dumps({'text': ' '.join(odd_input_ctx), 'label': ' '.join(odd_output_lbl)}, ensure_ascii=False) + "\n")
        output_f.write(json.dumps({'text': ' '.join(even_input_ctx), 'label': ' '.join(even_output_lbl)}, ensure_ascii=False) + "\n")
        odd_input_ctx, odd_output_lbl, even_input_ctx, even_output_lbl = [], [], [], []
        current_input_ctx_len, odd_extid_no, even_extid_no = 0, 0, 0

    last_sents = sents

output_f.close()
