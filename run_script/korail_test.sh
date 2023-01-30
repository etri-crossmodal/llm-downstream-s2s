TOKENIZERS_PARALLELISM=True CUDA_VISIBLE_DEVICES=2 python inference.py \
	-model korail_test_230130-kebyt5-base580m-preview0.3-adafactor/saved_checkpoints/epoch=13-step=1134_by_epoch.ckpt \
	-float_precision 16 -gpus 1 -batch_size 196 -task kr-internal

# mini - 74.0/adafactor
#-model korail_test_230130-kebyt5-mini-preview-adafactor/saved_checkpoints/epoch=23-step=3864_by_epoch.ckpt \
