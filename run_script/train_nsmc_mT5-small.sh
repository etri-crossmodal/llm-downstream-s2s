# nsmc-naive task. 별도의 프롬프트가 없음.
python train.py -task nsmc-naive \
	-save_path ./DELETEME-nsmc_test_mt5-small-baseline \
	-init_model "google/mt5-small" \
	-max_epoch 3 -learning_rate 1e-4 -gpus 2 \
       	-strategy ddp -float_precision 32 \
       	-grad_acc 4 -batch_size 4
