# NLI는 premise+hypothesis 결합, testset 최장길이 ~500 bytes
python train.py -task klue-nli \
  -save_path ./DELETEME_finetune_klue_nli_230807 \
  -init_model ../models/kebyt5-base-gbst_ds3x-preview-230627-ssm-stage2_230805/ \
  -strategy deepspeed_1 -float_precision 32 -max_epoch 4 -gpus 2 \
  -gradient_checkpointing 1 \
  -learning_rate 8e-5 -warmup_steps 0 -batch_size 1 -grad_acc 8 \
  -max_seq_length 768 -seed 591850 \
  -valid_check_interval 0.25 -optim_cosanneal_gamma 0.7 -optim_cosanneal_restarts 4 
