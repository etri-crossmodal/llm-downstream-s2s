python inference.py -test_data ../kojp_usuallife_validation.txt \
  -model ./kebyt5-small-kojp-trans-230220/saved_checkpoints/epoch\=3-step\=124028_by_epoch.ckpt \
  -tokenizer ../models/kebyt5-small-preview-230118/ \
  -gpus 1 -float_precision 32 \
  -save_output ../kojp_predict_out-small230220_fp32-beam5.txt -batch_size 256 \
  -beam_size 5 -max_predict_length 512
