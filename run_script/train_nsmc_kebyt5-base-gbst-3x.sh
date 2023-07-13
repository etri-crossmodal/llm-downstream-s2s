python train.py -task nsmc-prompted \
	-save_path ./DELETEME-nsmc_test_gbst-kebyt5-base-preview \
	-init_model "etri-lirs/gbst-kebyt5-base-preview" \
	-max_epoch 3 -learning_rate 8e-5 -gpus 2 \
       	-strategy ddp -float_precision 32 \
	-grad_acc 2 -batch_size 8 
