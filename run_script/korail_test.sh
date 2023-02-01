TOKENIZERS_PARALLELISM=True CUDA_VISIBLE_DEVICES=3 python inference.py \
  -model korail_test_230125-kebyt5-mini-preview/saved_checkpoints/epoch=31-step=5152_by_epoch.ckpt \
	-float_precision 16 -gpus 1 -batch_size 196 -task kr-internal

# mini - 74.0/adafactor
#-model korail_test_230130-kebyt5-mini-preview-adafactor/saved_checkpoints/epoch=23-step=3864_by_epoch.ckpt \
