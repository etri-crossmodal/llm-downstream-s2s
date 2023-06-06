python ./train.py -task klue-mrc \
  -save_path ./DELETEME_finetune_klue_mrc_230523 \
  -init_model ~/Works/models/kebyt5-large_230605_gs90.7k/ \
  -strategy deepspeed_3_full -float_precision 16 -max_epoch 4 -gpus 4 \
  -max_seq_length 2500 -seed 591850 \
  -learning_rate 6e-5 -warmup_steps 300 -batch_size 1 -grad_acc 3 \
  -valid_check_interval 0.25 -optim_cosanneal_gamma 0.7 -optim_cosanneal_restarts 4 

