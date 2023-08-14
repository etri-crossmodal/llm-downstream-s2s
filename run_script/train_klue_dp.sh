# NER task.  _gen_whole_sent 로 끝난 경로는 전체 문장을 복제하면서 생성
# 그렇지 않은 것은 <어휘:클래스> <어휘:클래스> ... 로 만 예측함
# collators/klue.py:177~178 을 참고, label texts에 뭐가 들어가는지, only_tags vs sentence로 이해하면 됨 
python train.py -task klue-dp \
  -save_path ./DELETEME_finetune_klue_dp_230811_input_lemma \
  -init_model etri-lirs/gbst-kebyt5-base-preview \
  -strategy deepspeed_1 -float_precision 16 -max_epoch 4 -gpus 4 \
  -gradient_checkpointing 0 \
  -learning_rate 8e-5 -warmup_steps 0 -batch_size 1 -grad_acc 4 \
  -max_seq_length 4096 -seed 591850 \
  -valid_check_interval 0.25 -optim_cosanneal_gamma 0.7 -optim_cosanneal_restarts 4 
