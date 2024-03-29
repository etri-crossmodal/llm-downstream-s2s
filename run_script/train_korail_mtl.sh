#python train.py -save_path ./korail_test_230125-kebyt5-mini-preview \
#	-init_model ../models/kebyt5-mini-preview-v1.05/ \
#	-max_epoch 32 -learning_rate 2e-4 \
#	-gpus 4 -strategy ddp -float_precision 32 \
#	-grad_acc 6  -batch_size 12 \
#	-task kr-internal \
#	-resume_checkpoint korail_test_230125-kebyt5-mini-preview/saved_checkpoints/epoch=27-step=4508_by_epoch.ckpt

#python train.py -save_path ./korail_test_230130-kebyt5-mini-preview-adafactor \
	#-init_model ../models/kebyt5-mini-preview-v1.05/ \
python train.py -save_path ./korail_test_230131-small-mtl-adafactor/ \
	-init_model ../models/kebyt5-small-preview-230118// \
  -max_epoch 16 -learning_rate 1e-3 \
  -gpus 1 -strategy ddp -float_precision 32 \
  -grad_acc 1  -batch_size 6 \
  -task kr-internal-mtl \
  -resume_checkpoint korail_test_230131-small-mtl-adafactor/saved_checkpoints/epoch=13-step=2041_by_epoch.ckpt
