#!/usr/bin/env python3
"""
    Export DeepSpeed 2/3 Checkpoint or lightning checkpoint to
    FP32, huggingface transformers-compatible downstream Model checkpoint.

    Copyright (C) 2023, Jong-hun Shin. ETRI LIRS.
"""

import sys
import os

from pathlib import Path

import torch

from models.mlm_plmodule_wrapper import ETRIT5ConditionalGenModelLightningModule


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("USAGE: python export_checkpoint_to_hfmodel.py "
              "[deepspeed checkpoint dirpath or lightning model checkpoint file] [output huggingface downstream model dirpath]")
        print("NOTE: Output pytorch model directory will be created automatically.")
        sys.exit(-1)

    interm_checkpoint_filename = sys.argv[2] + ".plmodel"
    if Path(sys.argv[1]).is_dir():
        # directory로 되어 있을 경우 DS2/3 -> HF model로
        ETRIT5ConditionalGenModelLightningModule.convert_deepspeed_checkpoint_to_fp32(sys.argv[1], interm_checkpoint_filename)
        model = ETRIT5ConditionalGenModelLightningModule.load_from_checkpoint(interm_checkpoint_filename, strict=False)
        model.export_hf_model(sys.argv[2])
        # remove intermediate pytorch-lightning checkpoint
        os.unlink(interm_checkpoint_filename)
    elif Path(sys.argv[1]).is_file():
        # DDP file이면 바로 HF 모델로
        model = ETRIT5ConditionalGenModelLightningModule.load_from_checkpoint(sys.argv[1], strict=False)
        model.export_hf_model(sys.argv[2])
