# 어차피 SSM으로 변환된 데이터는 1024~1280 바이트 이하.
python ./train.py -task hfdataset \
  -save_path ./gbst-kebyt5-base-aihub_ssm-230804-stage2 \
  -init_model ~/kebyt5-base-gbst_ds3x-preview-230627-ssm-aihub_mrc_ep1_230804/ \
  -train_data ~/Works/ptlm-downstream-test/salient_span_masked_training_data/aihub_mrcs_4_ssm_230804/ \
  -train_data ~/Works/ptlm-downstream-test/salient_span_masked_training_data/aihub_010_raw_문어_도서_SSM_230804/ \
  -valid_data ~/Works/ptlm-downstream-test/salient_span_masked_training_data/klue101_contextonly_ssm_230804/ \
  -strategy deepspeed_1 -float_precision 16 -max_epoch 4 -gpus 4 \
  -gradient_checkpointing 0 -max_seq_length 1280 -seed 123456 \
  -learning_rate 4e-5 -warmup_steps 0 -batch_size 8 -grad_acc 32 \
  -lr_scheduler linear \
  -valid_check_interval 0.1

# valid data를 klue로 세팅
#  -valid_data_proportions 0.05 \
