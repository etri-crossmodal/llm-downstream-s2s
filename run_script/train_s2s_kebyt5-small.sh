# -task seq2seq 학습 테스트. tabbed text 학습.
CUDA_VISIBLE_DEVICES=1,2,3 python ./train.py -task seq2seq \
  -save_path ./kebyt5-small-kojp-trans-230220 \
  -init_model ../models/kebyt5-small-preview-230118/ \
  -train_data ~/InternalCorpus/kojp-refined-190114-3.1M.txt \
  -valid_data_proportions 0.05 \
  -max_seq_length 1024 -grad_acc 2 -max_epoch 4 -batch_size 16 \
  -gpus 3 -float_precision 16 -optim adafactor

#-test_data ~/InternalCorpus/ \
