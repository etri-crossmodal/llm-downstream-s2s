python train.py -save_path ./nsmc_test_mt5-small-baseline -init_model "google/mt5-small" -max_epoch 4 -learning_rate 1e-4 -gpus 4 -strategy ddp -float_precision 16 -grad_acc 4 -batch_size 16
