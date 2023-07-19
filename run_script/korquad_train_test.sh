python ./train.py -task korquad-v1 \
  -save_path ./DELETEME_finetune_korquad_v1-kgbst6-base_230718 \
  -init_model etri-lirs/gbst-kebyt5-base-preview \
  -strategy deepspeed_1 -float_precision 32 -max_epoch 4 \
  -max_seq_length 4500 -seed 591850 -gradient_checkpointing 1 \
  -learning_rate 8e-5 -warmup_steps 300 -batch_size 1 -grad_acc 8 -gpus 2 \
  -valid_check_interval 0.25 -optim_cosanneal_gamma 0.7 -optim_cosanneal_restarts 4 

