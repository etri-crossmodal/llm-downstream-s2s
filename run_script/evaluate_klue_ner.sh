# 추론 예시.
CUDA_VISIBLE_DEVICES=1 python inference.py -task klue-ner \
	-model DELETEME_finetune_klue_ner_230808_gen_whole_sent/saved_checkpoints/last.ckpt \
	-gpus 1 -batch_size 4 \
	-max_seq_length 2048 -max_predict_length 3072 \
	-save_output gbst-base-ssm-stage2-ner-wholesent-predict-230809.txt \
	-save_label klue-ner-wholesent-gold.txt

# NER 평가. F1^e (entity-level F1)과, F1^c (char-level F1, KLUE는 IOB2 tag를 비교 했으므로, 동일하게 IOB2 태그로 변환 후 Word-level(=즉, 1 word=1 IOB2 tag) F1으로 계산함) 
# 별도로 label-only chrf(popovic et al., '15)를 추가로 계산함.
python tools/compute-metric-ner.py \
	gbst-base-ssm-stage2-ner-wholesent-predict-230809.txt \
	klue-ner-wholesent-gold.txt

