python train.py -save_path ./nsmc_test_et5-base -init_model ../models/et5-base/ -max_epoch 4 -learning_rate 1e-4 -gpus 4 -strategy ddp -float_precision 16 -grad_acc 2 -batch_size 32
