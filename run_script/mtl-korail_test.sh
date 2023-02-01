TOKENIZERS_PARALLELISM=True CUDA_VISIBLE_DEVICES=0 python inference.py \
  -model korail_test_230131-small-mtl-adafactor/saved_checkpoints/epoch=15-step=2063_by_epoch.ckpt \
	-float_precision 32 -gpus 1 -batch_size 196 -task kr-internal-mtl

# mini - 74.0/adafactor
#-model korail_test_230130-kebyt5-mini-preview-adafactor/saved_checkpoints/epoch=23-step=3864_by_epoch.ckpt \
