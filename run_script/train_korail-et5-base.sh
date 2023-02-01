#python train.py -save_path ./korail_test_230125-kebyt5-mini-preview \
#	-init_model ../models/kebyt5-mini-preview-v1.05/ \
#	-max_epoch 32 -learning_rate 2e-4 \
#	-gpus 4 -strategy ddp -float_precision 32 \
#	-grad_acc 6  -batch_size 12 \
#	-task kr-internal \
#	-resume_checkpoint korail_test_230125-kebyt5-mini-preview/saved_checkpoints/epoch=27-step=4508_by_epoch.ckpt
CUDA_VISIBLE_DEVICES=0 python train.py -save_path ./korail_test_230131-et5-base-mtl \
	-init_model ../models/et5-base/ \
  -max_epoch 6 -learning_rate 1e-3 \
  -gpus 1 -strategy ddp -float_precision 16 \
  -grad_acc 6  -batch_size 12 \
  -task kr-internal-mtl
