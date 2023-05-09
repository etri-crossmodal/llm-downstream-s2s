#!/usr/bin/env python3
"""
    Export DeepSpeed 2/3 Checkpoint or lightning checkpoint to
    FP32, huggingface transformers-compatible downstream Model checkpoint.

    Copyright (C) 2023, Jong-hun Shin. ETRI LIRS.
"""

import sys
import os

from pathlib import Path
from typing import Any

import torch

from models.mlm_plmodule_wrapper import ETRIT5ConditionalGenModelLightningModule


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("USAGE: python export_checkpoint_to_hfmodel.py "
              "[deepspeed checkpoint dirpath or lightning model checkpoint file] "
              "[output huggingface downstream model dirpath]")
        print("NOTE: Output pytorch model directory will be created automatically.")
        sys.exit(-1)

    interm_checkpoint_filename = sys.argv[2] + ".plmodel"
    if Path(sys.argv[1]).is_dir():
        # directory로 되어 있을 경우 DS2/3 -> HF model로
        try:
            ETRIT5ConditionalGenModelLightningModule.convert_deepspeed_checkpoint_to_fp32(
                sys.argv[1], interm_checkpoint_filename)
            model = ETRIT5ConditionalGenModelLightningModule.load_from_checkpoint(
                interm_checkpoint_filename, strict=False)
        except ValueError as e:
            if e.args[0] == "unknown zero stage 1":
                print("** Stage 1 checkpoint found, now trying to load state_dict from partitioned one.")
                interm_checkpoint_filename = ""

                # Monkey patching a LightningModule.on_load_checkpoint for deepspeed stage 1.
                def on_load_checkpoint(wouldbe_self, checkpoint: dict[str, Any]) -> None:
                    if "state_dict" in checkpoint:
                        return
                    state_dict = checkpoint['module']
                    state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
                    checkpoint['state_dict'] = state_dict
                    return

                ETRIT5ConditionalGenModelLightningModule.on_load_checkpoint = on_load_checkpoint
                model = ETRIT5ConditionalGenModelLightningModule.load_from_checkpoint(
                            args.checkpoint_path + "/checkpoint/mp_rank_00_model_states.pt",
                            strict=False)
            else:
                # 그 이외의 오류는 그대로 raise
                raise ValueError(e)

        model.export_hf_model(sys.argv[2])
        # remove intermediate pytorch-lightning checkpoint
        os.unlink(interm_checkpoint_filename)
    elif Path(sys.argv[1]).is_file():
        # DDP file이면 바로 HF 모델로
        model = ETRIT5ConditionalGenModelLightningModule.load_from_checkpoint(sys.argv[1], strict=False)
        model.export_hf_model(sys.argv[2])
