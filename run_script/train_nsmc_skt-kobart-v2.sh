python train.py -task nsmc-prompted \
	-save_path ./DELETEME-nsmc_test_skt-kobart-v2 \
	-init_model "gogamza/kobart-base-v2" \
	-max_epoch 3 -learning_rate 1e-4 -gpus 2 \
       	-strategy ddp -float_precision 32 \
	-grad_acc 2 -batch_size 8 
