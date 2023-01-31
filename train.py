#!/usr/bin/env python3
"""
    KEByT5 학습을 위한 메인 트레이너 루틴.

    need python 3.6 or more, tested under python 3.9

    Copyright (C) 2022~ ETRI Language Intelligence Research Section, Jong-hun Shin.
"""

import os
# Disable TF-TRT Warnings, we don't want to use tf2 for tensorboard.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import functools
import argparse

from dataclasses import dataclass
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from looseversion import LooseVersion
from torch import nn
from torch.utils.data import DataLoader

from torch.distributed.fsdp.wrap import (transformer_auto_wrap_policy)
from transformers.models.t5.modeling_t5 import T5Block

from transformers import (
    AutoConfig, BertModel,
    AutoTokenizer,
)
from datasets import (load_from_disk, load_dataset, DatasetDict)
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero

# we need pytorch 1.12+
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
#from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
#        FullStateDictConfig, StateDictType, MixedPrecision)


from models.mlm_plmodule_wrapper import ETRIT5ConditionalGenModelLightningModule

import task_utils

def get_argparser():
    """ generate argument parser. """
    parser = argparse.ArgumentParser(description="Train T5-like model with pytorch+transformers.")
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
    parser.add_argument("-max_seq_length", type=int, default=1024,
                        help="set maximum length of text in given datasets. "
                        "if example length exceeds, it will be discarded (WARNING: not truncated)")
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
    parser.add_argument("-save_last_k", type=int, default=25,
                        help="remain last k checkpoint.")
    parser.add_argument("-gpus", type=int, default=2,
                        help="number of accelerators(e.g. GPUs) for training.")
    parser.add_argument("-strategy", type=str, default="ddp",
                        help="DDP training strats. can be one of (fsdp_native_cpu_offload|deepspeed_2_optim_offload|deepspeed_3_full|ddp)")
    parser.add_argument("-float_precision", type=int, default=32,
                        help="set floating point precision. default value is 32, you can set 16. with value 16, if bf16 supported, bf16 will be enabled automatically.")
    parser.add_argument("-task", type=str, default="nsmc-prompted",
                        help="set a downstream task. (nsmc-naive|nsmc-prompted|klue-nli-prompted|translate-ko-en)")
    return parser


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    if args.config_path == "" and args.init_model == "":
        raise Exception("assign -config_path or -init_model to define a model. "
                        "e.g. -init_model google/byt5-small")
    elif args.config_path != "" and args.init_model != "":
        raise Exception("use -config_path or -init_model exclusively. do not use them both.")

    if args.seed < 0:
        # python 3.6 or more needed to use secrets
        import secrets
        args.seed = secrets.randbelow(1_000_000_000)

    # global seed 초기화
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    # add Tensorboard logger, w/o hp_metric
    logger = pl.loggers.TensorBoardLogger(base_dirpath,
            name="lightning_logs",
            default_hp_metric=False)

    # 기본적으로 Fused AdamW (DeepSpeed)는 Off, 32bit로 학습
    optimizer_arg = "adam"
    precision_arg = args.float_precision
    callbacks = []

    if precision_arg != 32 and precision_arg != 16:
        raise Exception("bad argument: you can assign 32 or 16 for -float_precision")

    bf16_ready = (torch.version.cuda and torch.cuda.is_bf16_supported()
            and LooseVersion(torch.version.cuda) >= "11.0"
            and torch.distributed.is_nccl_available())

    if bf16_ready and precision_arg == 32:
        print("NOTICE: This CUDA GPU supports bfloat16 precision. We suggest you use '-float_precision 16' for faster training.")
        # FIXME: avaliable if rank 0
        #input("Press Enter to continue...")

    # we should use CPU adamW for deepspeed
    if precision_arg == 16 and bf16_ready:
        print("** bfloat16 available: enable bfloat16 training, instead of fp16.")
        precision_arg = "bf16"

    if args.strategy == "fsdp_native_cpu_offload":
        raise Exception("ERROR: Not working properly, need to FIX; disabled for now.")
        print("** Strategy: fsdp_native+cpu offload")
        t5_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={ T5Block, },)
        fsdp_kwargs = { "auto_wrap_policy": t5_auto_wrap_policy, }

        strat_instance = DDPFullyShardedNativeStrategy(cpu_offload=CPUOffload(offload_params=True),
                                                       **fsdp_kwargs)
    elif args.strategy == "deepspeed_2_optim_offload":
        print("** Strategy: Microsoft DeepSpeed Zero 2 + Optimizer Offload")
        strat_instance = DeepSpeedStrategy(stage=2, remote_device="cpu",
                                           offload_optimizer=True, offload_parameters=False,
                                           pin_memory=True,
                                           allgather_bucket_size=2e8,
                                           reduce_bucket_size=2e8,
                                           contiguous_gradients=True, overlap_comm=True,
                                           logging_batch_size_per_gpu=args.batch_size)
        optimizer_arg = "cpuadam"
    elif args.strategy == "deepspeed_3_full":
        print("** Strategy: Microsoft DeepSpeed Zero 3 + Full(CPU) Offload")
        # _size=2e7, bf16, seq=1024, bs=4, grad_acc=64 ==> 3.7B(byt5-xl) not passed on 48GB VRAM (by OOM)
        # 학습에는 1 epoch에 130일 정도가 소요된다. 여전히 pressure가 너무 높아서 실 학습은 어려움
        # FIXME: VRAM 최적화를 위한 추가 조정 필요
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
        # downstream task이므로 adafactor를 사용
        optimizer_arg = "adafactor"

    # ================ FIXME for Training ==================
    data_module, collator, label_id_map = task_utils.get_task_data(args.task,
                                                                   args.batch_size,
                                                                   args.init_model)
    if data_module is None:
        raise Exception("invalid -task option argument.")
    # ======================================================

    model = ETRIT5ConditionalGenModelLightningModule(
        args.config_path, args.init_model,
        tokenizer=args.init_model,
        data_collator=collator,
        optimizer=optimizer_arg,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        train_batch_size=args.batch_size, val_batch_size=args.batch_size,
    )

    base_dirpath = args.save_path
    log_path = os.path.join(base_dirpath, "lightning_logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    checkpoint_dirpath = os.path.join(base_dirpath, "saved_checkpoints/")

    # add checkpoint saver
    # 이건 gradient acc때문임: every_n_train_steps * gradient_acc = 실제 저장되는 시점
    # FIXME: every_n_train_steps를 외부에서 결정할 수 있게.
    if args.save_every > 0:
        checkpoint_cb = ModelCheckpoint(dirpath=checkpoint_dirpath,
                filename='{epoch}-{step}',
                save_top_k=args.save_last_k,
                # 'global_step'은 매뉴얼과 달리 올바르게 monitoring 되지 않는다. 사용하지 말것
                monitor="step",
                mode="max",
                every_n_train_steps=args.save_every,  # -save_every(=150) * grad_acc(=32) = save checkpoints every 4800 steps
                every_n_epochs=None,        # every_n_train_steps/every_n_epochs/train_time_interval must be exclusive
                train_time_interval=None,
                verbose=True)
        callbacks.append(checkpoint_cb)

    checkpoint_cb_by_ep = ModelCheckpoint(dirpath=checkpoint_dirpath,
            filename='{epoch}-{step}_by_epoch',
            every_n_train_steps=None,
            every_n_epochs=1,
            save_top_k=-1,
            train_time_interval=None,
            verbose=True)
    callbacks.append(checkpoint_cb_by_ep)

    # add Learning Rate Monitor - FIXME: not working
    lr_mon = LearningRateMonitor(logging_interval=None)
    callbacks.append(lr_mon)

    # initialize trainer,
    # fsdp_native를 사용해야 할 경우, configure_optimizer에서
    # torch.optim.AdamW()에 self.trainer.model.parameters()가 전달되어야 함.
    # bf16을 쓰고 싶으면 ampere급 이상 GPU가 있는 곳에서 해라. 최소 A6000 필요함
    # 호스트 메모리가 적으면 nvme offloading도 고려해야 함
    # gradient accumulation을 사용하면 global_step이 그에 맞게 떨어진다.
    # learning rate scheduler를 위해서, max_epoch을 충분히 크게 잡을 것.

    trainer = pl.Trainer(accelerator="gpu",
            devices=args.gpus,
            callbacks=callbacks,
            default_root_dir=base_dirpath,
            logger=logger,
            check_val_every_n_epoch=1,
            val_check_interval=1.0,
            max_epochs=args.max_epoch,
            log_every_n_steps=1,
            accumulate_grad_batches=args.grad_acc,
            precision=precision_arg,
            strategy=strat_instance
            )

    print("** Model Fitting Started.")
    if args.resume_checkpoint != "":
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)

    # 실험을 중단해야 할 때 마지막으로 사용하는 것. Deepspeed 사용 시 신뢰해서는 안됨
    print("Now we trying to save fallback checkpoint to 'save_checkpoint_fallback.pt'. please wait for a while")
    trainer.save_checkpoint(base_dirpath + "/save_checkpoint_fallback.pt")
    print("Fallback checkpoint was saved.")
