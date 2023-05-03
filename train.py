#!/usr/bin/env python3
"""
    KEByT5 학습을 위한 메인 트레이너 루틴.

    need python 3.6 or more, tested under python 3.9

    Copyright (C) 2022~ ETRI Language Intelligence Research Section, Jong-hun Shin.
"""
import os
import random
import functools
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
import task_utils

from datetime import timedelta
from torch.distributed.fsdp.wrap import (transformer_auto_wrap_policy)
from transformers.models.t5.modeling_t5 import T5Block
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# we need pytorch 1.12+
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from packaging.version import Version


from models.mlm_plmodule_wrapper import ETRIT5ConditionalGenModelLightningModule


# Disable TF-TRT Warnings, we don't want to use tf2 for tensorboard.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_argparser():
    """ generate argument parser. """
    parser = argparse.ArgumentParser(description="Train T5-like model with pytorch+transformers.")
    parser.add_argument("-task", type=str, default="seq2seq",
                        help="set a downstream task. (nsmc-naive|nsmc-prompted|"
                             "klue-nli-prompted|translate-ko-en)")

    parser.add_argument("-train_data", type=str, action='append', required=False,
                        help="must be provided when you use -task seq2seq option. you can assign "
                        "huggingface dataset name or dataset path, which reserved by "
                        "datasets.save_to_disk(), and tabbed text file(.txt|.tsv). "
                        "you can assign multiple dataset by repeating -data option, "
                        "they will be concatenated into single dataset. "
                        "and you can shard a dataset then learn just 1/N samples"
                        "by appending ':N(N=number)' as suffix.")
    parser.add_argument("-valid_data", type=str, action='append', required=False,
                        help="same as -train_data option.")
    parser.add_argument("-test_data", type=str, action='append', required=False,
                        help="same as -train_data option.")
    parser.add_argument("-valid_data_proportions", type=float, default=0.0,
                        help="when -task seq2seq and no -valid_data, we will create validation data from "
                        "train data.")
    parser.add_argument("-test_data_proportions", type=float, default=0.0,
                        help="when -task seq2seq and no -valid_data, we will create validation data from "
                        "train data.")
    parser.add_argument("-max_seq_length", type=int, default=0,
                        help="set maximum token length of text in given datasets. "
                        "if example length exceeds, it will be DISCARDED without -do_truncate=True)")
    parser.add_argument("-do_truncate", type=bool, default=True,
                        help="If it sets to TRUE, truncate input(not label!) text with max_seq_length. "
                        "default bahavior=FALSE=just discard when exceeds max_seq_length. "
                        "however, when label exceeds max_seq_length, we will discard it whatsoever.")

    parser.add_argument("-seed", type=int, default=123456,
                        help="set a seed for RNGs. if you assign value below 0(e.g. -1), "
                        "we will randomize seed with secrets.randbelow() function.")
    parser.add_argument("-batch_size", type=int, default=48,
                        help="train/valid data batch size")
    parser.add_argument("-config_path", type=str, default="",
                        help="use only when you want to train from scratch, "
                        "or resume training with -resume_checkpoint.")
    parser.add_argument("-init_model", type=str, default="",
                        help="use only when you want to train from "
                        "another pretrained model. e.g. google/byt5-small")
    parser.add_argument("-resume_checkpoint", type=str, default="",
                        help="resume training with given checkpoint directory.")
    parser.add_argument("-grad_acc", type=int, default=1,
                        help="gradient accumulation to increase effective batch size.")
    parser.add_argument("-max_epoch", type=int, default=4,
                        help="set maximum epoch limit. this value controls NOAM scheduler. "
                        "DO NOT set to -1(infinite) or >=100, learning rate will not decay properly.")
    parser.add_argument("-learning_rate", type=float, default=1e-4,
                        help="set peak learning rate. see also -warmup_step")
    parser.add_argument("-warmup_steps", type=int, default=0,
                        help="set warmup steps for NOAM scheduler. "
                        "if you want to train from scratch, you must assign the value with >1000 steps")
    parser.add_argument("-save_path", type=str, required=True,
                        help="set model/log save path.")
    parser.add_argument("-save_every", type=int, default=0,
                        help="save every n global steps. default: 0 (disable)"
                        "WARNING: -save_every=k * -grad_acc=x = save checkpoints every k*x steps")
    parser.add_argument("-save_every_hour", type=int, default=1,
                        help="save checkpoint in every N hour.")
    parser.add_argument("-save_last_k", type=int, default=2,
                        help="remain last k checkpoint. if you want to save checkpoint before validation, "
                        "please set this value to 0, or you will lose some early checkpoints.")
    parser.add_argument("-valid_check_interval", type=float, default=1.0,
                        help="set validation check interval, 1.0 = end of an epoch, 0.5 = half of an epoch.")
    parser.add_argument("-gpus", type=int, default=2,
                        help="number of accelerators(e.g. GPUs) for training.")
    parser.add_argument("-strategy", type=str, default="ddp",
                        help="DDP training strats. can be one of (fsdp_native_cpu_offload|"
                        "deepspeed_2_optim_offload|deepspeed_3_full|ddp)")
    parser.add_argument("-float_precision", type=int, default=32,
                        help="set floating point precision. default value is 32, "
                        "you can set 16, and if bf16 supported, bf16 will be enabled automatically.")
    parser.add_argument("-optim", type=str, default="adam",
                        help="set optimizer. with adafactor, recommend to use -learning_rate 0.001")
    parser.add_argument("-tuning_method", type=str, default="finetune",
                        help="EXPERIMENTAL: use for parameter-efficient fine-tuning."
                        "you can use one of [lora/prefixtuning/finetune]")
    parser.add_argument("-gradient_checkpointing", type=int, default=0,
                        help="Enable Gradient checkpointing. you can use it when you suffering from OOM.")
    return parser


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    if args.task == "seq2seq" and (args.train_data is None or len(args.train_data) == 0):
        raise ValueError("you must assign -data option when -task seq2seq.")

    if args.config_path == "" and args.init_model == "":
        raise ValueError("assign -config_path or -init_model to define a model. "
                        "e.g. -init_model google/byt5-small")
    elif args.config_path != "" and args.init_model != "":
        raise ValueError("use -config_path or -init_model exclusively. do not use them both.")

    if args.valid_check_interval < 0.0 or args.valid_check_interval > 1.0:
        raise ValueError("-valid_check_interval must be in [0.0, 1.0]")

    if args.seed < 0:
        # python 3.6 or more needed to use secrets
        import secrets
        args.seed = secrets.randbelow(1_000_000_000)

    grad_checkpointing = False
    if args.gradient_checkpointing != 0:
        grad_checkpointing = True

    # global seed 초기화
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    accelerator_args = "gpu"
    accelerator_counts = args.gpus

    if args.gpus <= 0:
        accelerator_args = "cpu"
        accelerator_counts = int(len(os.sched_getaffinity(0)) // 1.5)

    base_dirpath = args.save_path
    log_path = os.path.join(base_dirpath, "lightning_logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # huggingface datasets를 위한 cache_dir을 model-specific하게 다룰 수 있도록 함.
    hf_cache_path = os.path.join(base_dirpath, "hf_datasets_cache")
    if not os.path.exists(hf_cache_path):
        os.makedirs(hf_cache_path)

    checkpoint_dirpath = os.path.join(base_dirpath, "saved_checkpoints/")

    # add Tensorboard logger, w/o hp_metric
    logger = pl.loggers.TensorBoardLogger(base_dirpath,
            name="lightning_logs",
            default_hp_metric=False)

    # 기본적으로 Fused AdamW (DeepSpeed)는 Off, 32bit로 학습
    optimizer_arg = args.optim
    precision_arg = args.float_precision
    callbacks = []

    if precision_arg != 32 and precision_arg != 16:
        raise ValueError("bad argument: you can assign 32 or 16 for -float_precision")

    bf16_ready = (torch.version.cuda and torch.cuda.is_bf16_supported()
            and Version(torch.version.cuda) >= Version("11.0")
            and torch.distributed.is_nccl_available())

    if bf16_ready and precision_arg == 32:
        print("NOTICE: This CUDA GPU supports bfloat16 precision. "
              "We suggest you use '-float_precision 16' for faster training.")
        # FIXME: avaliable if rank 0
        #input("Press Enter to continue...")

    # we should use CPU adamW for deepspeed
    if precision_arg == 16 and bf16_ready:
        print("** bfloat16 available: enable bfloat16 training, instead of fp16.")
        precision_arg = "bf16"

    if args.strategy == "fsdp_native_cpu_offload":
        raise NotImplementedError("ERROR: Not working properly, need to FIX; disabled for now.")
        print("** Strategy: fsdp_native+cpu offload")
        t5_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={ T5Block, },)
        fsdp_kwargs = { "auto_wrap_policy": t5_auto_wrap_policy, }

        strat_instance = DDPFullyShardedNativeStrategy(cpu_offload=CPUOffload(offload_params=True),
                                                       **fsdp_kwargs)
    elif args.strategy == "deepspeed_1":
        # just partition optimizer states, so we can use any optimizers with it.
        strat_instance = DeepSpeedStrategy(stage=1, remote_device="gpu",
                                           reduce_bucket_size=2e8,
                                           logging_batch_size_per_gpu=args.train_batch_size)
    elif args.strategy == "deepspeed_2_fusedadam":
        # uses more VRAM ~5GB from deepspeed_2_optim_offload + cpuadam.
        optimizer_arg = "fusedadam"
        strat_instance = DeepSpeedStrategy(stage=2, remote_device="gpu",
                                           offload_optimizer=False, offload_parameters=False,
                                           pin_memory=True,
                                           allgather_bucket_size=2e8,
                                           reduce_bucket_size=2e8,
                                           contiguous_gradients=True, overlap_comm=True,
                                           partition_activations=True, cpu_checkpointing=True,
                                           logging_batch_size_per_gpu=args.train_batch_size)
    elif args.strategy == "deepspeed_2_optim_offload":
        print("** Strategy: Microsoft DeepSpeed Zero 2 + Optimizer Offload")
        optimizer_arg = "cpuadam"
        strat_instance = DeepSpeedStrategy(stage=2, remote_device="cpu",
                                           offload_optimizer=True, offload_parameters=False,
                                           pin_memory=True,
                                           allgather_bucket_size=2e8,
                                           reduce_bucket_size=2e8,
                                           contiguous_gradients=True, overlap_comm=True,
                                           logging_batch_size_per_gpu=args.batch_size)
    elif args.strategy == "deepspeed_3_full":
        print("** Strategy: Microsoft DeepSpeed Zero 3 + Full(CPU) Offload")
        # _size=2e7, bf16, seq=1024, bs=4, grad_acc=64 ==> 3.7B(byt5-xl) not passed on 48GB VRAM (by OOM)
        # 학습에는 1 epoch에 130일 정도가 소요된다. 여전히 pressure가 너무 높아서 실 학습은 어려움
        # FIXME: VRAM 최적화를 위한 추가 조정 필요
        optimizer_arg = "cpuadam"
        strat_instance = DeepSpeedStrategy(stage=3, remote_device="cpu",
                                           offload_optimizer=True, offload_parameters=True,
                                           allgather_bucket_size=1.3e8,
                                           reduce_bucket_size=1.3e8,
                                           pin_memory=True,
                                           contiguous_gradients=True, overlap_comm=True,
                                           sub_group_size=1e8,
                                           partition_activations=True, cpu_checkpointing=True,
                                           logging_batch_size_per_gpu=args.batch_size)
    else:
        if args.strategy != "ddp":
            print("** Unknown strategy: set to ddp.")
        strat_instance = "ddp"

    # ================ FIXME for Training ==================
    # FIXME: task config를 별도로 두도록 하여 인자 수를 간소화하고,
    # 확장성을 확보해야 함
    data_module, collator, label_id_map = task_utils.get_task_data(
        args.task, args.batch_size, args.init_model,
        args.train_data, args.valid_data, args.test_data,
        args.valid_data_proportions, args.test_data_proportions,
        args.max_seq_length, args.do_truncate,
        hf_cache_path,
    )

    if data_module is None:
        raise ValueError("invalid -task option argument.")
    # ======================================================

    model = ETRIT5ConditionalGenModelLightningModule(
        args.config_path, args.init_model,
        tokenizer=args.init_model,
        data_collator=collator,
        optimizer=optimizer_arg,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        train_batch_size=args.batch_size, val_batch_size=args.batch_size,
        tuning_method=args.tuning_method,
        gradient_checkpointing=grad_checkpointing,
    )

    # add checkpoint saver
    # 이건 gradient acc때문임: every_n_train_steps * gradient_acc = 실제 저장되는 시점
    if args.save_every > 0:
        checkpoint_cb = ModelCheckpoint(
            dirpath=checkpoint_dirpath,
            filename='epoch{epoch:02d}-global_step{step}-val_loss{val_loss:.2f}',
            monitor="val_loss",
            verbose=True,
            auto_insert_metric_name=False,
            # 'global_step'은 매뉴얼과 달리 올바르게 monitoring 되지 않는다. 사용하지 말것
            mode="min",
            # -save_every(=150) * grad_acc(=32) = save checkpoints every 4800 steps
            save_top_k=args.save_last_k,
            every_n_train_steps=args.save_every,
            # every_n_train_steps/every_n_epochs/train_time_interval must be exclusive
            every_n_epochs=None,
            train_time_interval=None,
        )
        callbacks.append(checkpoint_cb)

    # epoch-wise checkpoint saving
    checkpoint_cb_by_ep = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename='epoch{epoch:02d}-global_step{step}-val_loss{val_loss:.2f}_endofepoch',
        monitor="val_loss",
        verbose=True, save_last=True,
        mode="min",
        auto_insert_metric_name=False,
        every_n_train_steps=None,
        every_n_epochs=1,
        save_top_k=-1,
        train_time_interval=None,
    )
    callbacks.append(checkpoint_cb_by_ep)

    # time-based checkpoint saving
    if args.save_every_hour > 0:
        delta = timedelta(
                days=0,
                seconds=0,
                microseconds=0,
                milliseconds=0,
                minutes=0,
                hours=args.save_every_hour,
                weeks=0)

        checkpoint_cb_by_hours = ModelCheckpoint(
            dirpath=checkpoint_dirpath,
            monitor='step',
            mode="max",
            filename='epoch{epoch:02d}-global_step{step}-by-hour',
            auto_insert_metric_name=False,
            every_n_train_steps=None,
            every_n_epochs=None,
            save_top_k=2,
            train_time_interval=delta,
            verbose=True)
        callbacks.append(checkpoint_cb_by_hours)

    # add Learning Rate Monitor
    lr_mon = LearningRateMonitor(logging_interval=None)
    callbacks.append(lr_mon)

    # initialize trainer,
    # fsdp_native를 사용해야 할 경우, configure_optimizer에서
    # torch.optim.AdamW()에 self.trainer.model.parameters()가 전달되어야 함.
    # bf16을 쓰고 싶으면 ampere급 이상 GPU가 있는 곳에서 해라. 최소 A6000 필요함
    # 호스트 메모리가 적으면 nvme offloading도 고려해야 함
    # gradient accumulation을 사용하면 global_step이 그에 맞게 떨어진다.
    # learning rate scheduler를 위해서, max_epoch을 충분히 크게 잡을 것.

    trainer = pl.Trainer(accelerator=accelerator_args,
            devices=accelerator_counts,
            callbacks=callbacks,
            default_root_dir=base_dirpath,
            logger=logger,
            check_val_every_n_epoch=1,
            val_check_interval=args.valid_check_interval,
            max_epochs=args.max_epoch,
            log_every_n_steps=1,
            accumulate_grad_batches=args.grad_acc,
            precision=precision_arg,
            strategy=strat_instance
            )

    # first validation for -save_every
    if args.save_every > 0:
        print("** Validate Initialized Model before fitting for checkpointing functionality.")
        trainer.validate(model=model, dataloaders=data_module)

    print("** Model Fitting Started.")
    if args.resume_checkpoint != "":
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)

    # 실험을 중단해야 할 때 마지막으로 사용하는 것. Deepspeed 사용 시 신뢰해서는 안됨
    if trainer.state.status == "interrupted":
        print("Now we trying to save fallback checkpoint to 'save_checkpoint_fallback.pt'. "
              "please wait for a while")
        trainer.save_checkpoint(base_dirpath + "/save_checkpoint_fallback.pt")
        print("Fallback checkpoint was saved.")
    else:
        if args.tuning_method != 'finetune':
            # PEFT를 썼으면 바로 모델을 export 할 것.
            print("Trained With Parameter-Efficient Fine-Tuning, save adapter checkpoint.")
            model.export_hf_model(args.save_path.rstrip('/\\') + '_adapter_ckpt')
        else:
            model.export_hf_model(args.save_path.rstrip('/\\') + '_hfmodel')
