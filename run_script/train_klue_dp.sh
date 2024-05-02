# NER task.  _gen_whole_sent 로 끝난 경로는 전체 문장을 복제하면서 생성
# 그렇지 않은 것은 <어휘:클래스> <어휘:클래스> ... 로 만 예측함
# collators/klue.py:177~178 을 참고, label texts에 뭐가 들어가는지, only_tags vs sentence로 이해하면 됨 
python train.py -task klue-dp \
  -save_path ./DELETEME_finetune_klue_dp_ds6x-large-406k-lr6e-5-240502 \
  -init_model ~/gbst-large-ds6x-newblock-406k-240501/ \
  -strategy deepspeed_1 -float_precision 16 -max_epoch 4 -gpus 2 \
  -gradient_checkpointing 0 \
  -learning_rate 5.8e-5 -warmup_steps 0 -batch_size 2 -grad_acc 4 \
  -max_seq_length 5000 -seed 591850 \
  -valid_check_interval 1.0 -optim_cosanneal_gamma 0.7 -optim_cosanneal_restarts 4 

#  -init_model etri-lirs/gbst-kebyt5-base-preview
CUDA_VISIBLE_DEVICES=1 python inference.py -task klue-dp \
	-model DELETEME_finetune_klue_dp_ds6x-large-406k-lr6e-5-240502/saved_checkpoints/last.ckpt \
	-max_seq_length 6000 -max_predict_length 8192 -batch_size 4 \
	-gpus 1 \
	-save_output ./klue_dp_pred_out_gbst-large-406k-lr6e-5-v3d-240502.txt \
	-save_label ./klue_dp_gold_out-v3d.txt
