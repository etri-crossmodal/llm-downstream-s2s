# llm-downstream-s2s
Trains downstream task (e.g. label prediction, translation, summary text generation, ...)  with encoder-decoder based Pre-trained Language Model

Supports following pretrained checkpoints, and also supports other encoder-decoder models (e.g. BART, T5):
  * etri-lirs/kebyt5-small-preview
  * etri-lirs/kebyt5-base-preview
  * etri-lirs/kebyt5-large-preview
  * etri-lirs/gbst-kebyt5-base-preview
  * etri-lirs/gbst-kebyt5-large-preview (not public released yet, soon be available.)

Copyright (C), 2023- Electronics and Telecommunications Research Institute. All rights reserved.


## How To Use

### Create Environments with conda (optional, can be replaced with pip)
Python environment preparation with conda. (see https://github.com/conda-forge/miniforge, or anaconda distribution as your needs)

```
$ conda create -n kebyt5-dev python=3.9
$ conda activate kebyt5-dev
$ conda env update -n kebyt5-dev  --file conda-environment.yaml
```

or you can use requirements.txt with python pip.


### Execute Trainer
```
python train.py -task seq2seq \
-train_data [training data text path, input-label must be paired in a line as tab-separated text format (TSV).] \
{ -valid_data [optional, validation data text path] } \
-valid_data_proportions [defaults: 0.05] \
-save_path [output checkpoint save path. directory will be created automatically.] \
-init_model [initial pre-trained model file path, or huggingface-compatible model name, e.g. etri-lirs/kebyt5-base-preview] \
-max_epoch [default: 4] \
-optim [optimizer; can be one of 'adam', 'adafactor', 'cpuadam', 'adam8'] \
-learning_rate [maximum learning rate. e.g. 8e-5] \
-gpus [counts of the gpu] \
-strategy [default: ddp, can be one of 'ddp', 'deepspeed_1', 'deepspeed_2_fusedadam', 'deepspeed_2_optim_offload', ...] \
-float_precision [defaults: 32, you can assign 16 to use bf16(if supported), or fp16.] \
-grad_acc [default: 1, # of gradient accumulation to increase effective batch size.] \
-batch_size [default: 16, batch size per gpu or device.] \
-tuning_method [default: 'finetune', you can use 'lora' with PEFT library.]

```	

if you want to resume training from last checkpoint, execute trainer with ``-resume_checkpoint [checkpoint path]`` option.


### Execute Inference Test
```bash
python inference.py -task seq2seq \
-test_data [file name of test data.] \
-model [trained checkpoint file, ends with .ckpt extension, or deepspeed checkpoint path.] \
-tokenizer [tokenizer path or huggingface-compatible model name, e.g. google/byt5-small] \
-gpus 1 -float_precision 32 \
-save_output [file name to save output results] -batch_size 64 \
-beam_size [beam size, default: 1.] \
-max_predict_length [maximum generation token length, default: 512 bytes for byt5 model.]
```


### Convert lightning checkpoints to huggingface model

```bash
python export_checkpoint_to_hfmodel.py [checkpoint directory/file path] [output huggingface model path]
```


#### seq2seq inference test with huggingface model

```bash
python hfmodel_s2s_inference.py \
-m [model file path, or huggingface-compatible model name] \
{-a [adapter model file path]} \
{-t [tokenizer path, or name, when model has not tokenizer configuration]} \
{-i [input text filename, which consists of a single input line. if it is not given from option, STDIN will be used.]} \
{-o [output text filename, or STDOUT will be used.]}
```


#### Execution Examples
As an example of execution, the nsmc classification test sample is included in run_scripts.
   * If you want to use SKT kobart-v2:, see train_nsmc_skt-kobart-v2.sh file.
   * If you have a kebyt5-* model, you can check the train_nsmc_kebyt5-small.sh file accordingly.

Please refer to the nsmc_test.sh file for inference examples. For training and evaluation of tab-delimited data, please refer to the following script:
   * Training - run_script/train_s2s_kebyt5-small.sh
   * Evaluation (inference) - run_script/inference_s2s_kebyt5-small.sh


## Dependencies
 * pytorch>=1.8.0
 * pytorch-lightning>=1.9.0
 * transformers>=4.27.0
 * einops>=0.6.0
 * evaluate
 * datasets
 * deepspeed

see requirements.txt


## Acknowledgement

 * This software was supported by the Institute of Information & communication Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT). (No. RS-2022-00187238, Development of Large Korean Language Model Technology for Efficient Pre-training)
 * This software contains code of Cosine-Annealing with Warm-up LR Scheduler (in models.mlm_plmodule_wrapper.py file) implementation, which derived from katsura-jp/pytorch-cosine-annealing-with-warmup (https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup) github project, which distributed under MIT License. Copyright (C) 2022 Naoki Katsura. All Rights Reserved.
 * This software includes lucidrains/charformer-pytorch GitHub project for GBST implementation, which distributed under MIT License. Copyright (c) 2021 Phil Wang. all rights reserved. (Original Code URL: https://github.com/lucidrains/charformer-pytorch)
 * This software includes HuggingFace transformers's T5 implementation for GBST-enabled T5 model, which distributed under Apache 2.0 License. Copyright 2018- The Huggingface team. All rights reserved.

     We are grateful for their excellent works.

