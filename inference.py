import os
# Disable TF-TRT Warnings, we don't want to use tf2 for tensorboard.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import functools
import argparse

from collections import Counter
from dataclasses import dataclass
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import evaluate
import jellyfish

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

from models import test_helper
from models.mlm_plmodule_wrapper import ETRIT5ConditionalGenModelLightningModule
from datamodules.nsmc_pldm import NSMCDataModule
from datamodules.klue_nli_pldm import KLUENLIDataModule, KLUEYNATDataModule
from datamodules.kornli_pldm import KorNLIDataModule
from datamodules.pawsx_pldm import paws_xDataModule
from datamodules.kortrain_test import korTrainTextDataModule

from collators import (generic, klue, pawsx)

def get_argparser():
    """ generate argument parser. """
    parser = argparse.ArgumentParser(description="Train T5-like model with pytorch+transformers.")
    parser.add_argument("-tokenizer", type=str, default="google/byt5-small",
                        help="set hf tokenizer name or model path.")
    parser.add_argument("-seed", type=int, default=123456,
                        help="set a seed for RNGs. if you assign value below 0(e.g. -1), "
                        "we will randomize seed with secrets.randbelow() function.")
    parser.add_argument("-batch_size", type=int, default=128,
                        help="train/valid data batch size")
    parser.add_argument("-model", type=str, default="",
                        help="model path or hf-model name. e.g. google/byt5-small")
    parser.add_argument("-gpus", type=int, default=4,
                        help="number of accelerators(e.g. GPUs) for training.")
    parser.add_argument("-float_precision", type=int, default=32,
                        help="set floating point precision. default value is 32, you can set 16. with value 16, if bf16 supported, bf16 will be enabled automatically.")
    parser.add_argument("-task", type=str, default="nsmc-prompted",
                        help="set a downstream task. (nsmc-naive|nsmc-prompted|klue-nli-prompted|translate-ko-en)")
    return parser


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    if args.model == "":
        raise Exception("assign -model to inference a model. "
                        "e.g. -model google/byt5-small")

    if args.seed < 0:
        # python 3.6 or more needed to use secrets
        import secrets
        args.seed = secrets.randbelow(1_000_000_000)

    # global seed 초기화
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    # 기본적으로 Fused AdamW (DeepSpeed)는 Off, 32bit로 학습
    use_cpu_adam_arg = False
    precision_arg = args.float_precision
    callbacks = []

    if precision_arg != 32 and precision_arg != 16:
        raise Exception("bad argument: you can assign 32 or 16 for -float_precision")

    bf16_ready = (torch.version.cuda and torch.cuda.is_bf16_supported()
            and LooseVersion(torch.version.cuda) >= "11.0"
            and torch.distributed.is_nccl_available())

    if bf16_ready and precision_arg == 32:
        print("NOTICE: This CUDA GPU supports bfloat16 precision. We suggest you use '-float_precision 16' for faster inference.")
        #input("Press Enter to continue...")

    # we should use CPU adamW for deepspeed
    if precision_arg == 16 and bf16_ready:
        print("** bfloat16 available: enable bfloat16 training, instead of fp16.")
        precision_arg = "bf16"

    # ================ FIXME for Training ==================
    if args.task == "nsmc-naive":
        # NSMC - naive version
        data_module = NSMCDataModule(batch_size=args.batch_size)
        collator = generic.GenericDataCollator(input_field_name="document",
                                               label_field_name="label",
                                               tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
                                               label_map={0: 'positive', 1: 'negative'})
    elif args.task == "nsmc-prompted":
        data_module = NSMCDataModule(batch_size=args.batch_size)
        collator = generic.GenericPromptedDataCollator(input_field_name="document",
                label_field_name="label",
                input_template="nsmc sentiment classification: {{ input }}",
                label_template="{{ label }}",
                tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
                label_map={0:'positive', 1:'negative'})
    elif args.task == "klue-nli-prompted":
        # Example 2: KLUE-NLI
        data_module = KLUENLIDataModule(batch_size=args.batch_size)
        collator = klue.KLUENLIDataCollator(tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),)
    elif args.task == "klue-ynat-prompted":
        # Example: KLUE-YNAT
        data_module = KLUEYNATDataModule(batch_size=args.batch_size)
        collator = klue.KLUEYNATDataCollator(tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),)
    elif args.task == 'kornli-prompted':
        # Example 3: KorNLI
        data_module = KorNLIDataModule(batch_size=args.batch_size)
        collator = klue.KLUENLIDataCollator(tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),)
    elif args.task == 'paws-x-kor':
        data_module = paws_xDataModule(batch_size=args.batch_size)
        collator = pawsx.PAWS_XDataCollator(tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),)
    elif args.task == 'kr-internal':
        # Korail, Internal Dataset, Multiclass classification problem.
        data_module = korTrainTextDataModule(batch_size=args.batch_size)
        collator = generic.GenericPromptedDataCollator(input_field_name="title",
                label_field_name="label",
                input_template="Classification Test:\n\n{{ input }}",
                label_template="{{ label }}",
                tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
                # Just pass text label into..
                label_map=data_module.id_to_label_func())
    elif args.task == "ko-en-translate":
        # Example 3: Translation
        """
        data_module = ...
        collator = ...
        """
        raise NotImplemented
    else:
        print('-task option now supported on: nsmc-naive, nsmc-prompted, klue-nli-prompted.')
        raise NotImplemented
    # ======================================================

    model = ETRIT5ConditionalGenModelLightningModule.load_from_checkpoint(args.model)
    model.tknizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # set collator
    model.data_collator = collator

    # initialize trainer,
    # fsdp_native를 사용해야 할 경우, configure_optimizer에서
    # torch.optim.AdamW()에 self.trainer.model.parameters()가 전달되어야 함.
    # bf16을 쓰고 싶으면 ampere급 이상 GPU가 있는 곳에서 해라. 최소 A6000 필요함
    # 호스트 메모리가 적으면 nvme offloading도 고려해야 함
    # gradient accumulation을 사용하면 global_step이 그에 맞게 떨어진다.
    # learning rate scheduler를 위해서, max_epoch을 충분히 크게 잡을 것.

    trainer = pl.Trainer(accelerator="gpu",
            devices=1, num_nodes=1,
            precision=precision_arg,
            )
    trainer.test(model, datamodule=data_module)

    # correct_pred_labels는 반드시 dict[str, int]가 되어야 함
    if args.task == "klue-nli-prompted" or args.task == "kornli-prompted":
        correct_pred_labels = {"entailment":0, "neutral":1, "contradiction":2}
    elif args.task == "nsmc-naive" or args.task == "nsmc-prompted":
        correct_pred_labels = {"positive":0, "negative":1}
    elif args.task == "paws-x-kor":
        correct_pred_labels = {"different":0, "paraphrase":1}
    elif args.task == "klue-ynat-prompted":
        correct_pred_labels = {'IT/science':0, 'economy':1, 'social':2,
                               'life and culture':3, 'world':4, 'sports':5, 'politics':6}
    elif args.task == 'kr-internal':
        correct_pred_labels = data_module.label_to_id_map_dict()
    else:
        raise NotImplemented

    print(f"# Test Labels: {len(test_helper.INFER_LABELS)}")
    print(f"# Test Predictions: {len(test_helper.INFER_PREDICTIONS)}")

    print(f"Predicted Unique labels(will include mis-typed label elements):")
    cnts = Counter(test_helper.INFER_PREDICTIONS)
    for k, v in cnts.items():
        print(f"\tlabel text: [{k}], counts: {v}")

    print("\n* trying to correct mis-typed labels with levenshtein(edit) distance.")
    correct_map = {}
    for k in cnts.keys():
        minimum_dist = 10000
        lowest_dist_lbl = ''
        for c in correct_pred_labels.keys():
            ck_dist = jellyfish.levenshtein_distance(c, k)
            if minimum_dist > ck_dist:
                lowest_dist_lbl = c
                minimum_dist = ck_dist
        if k != lowest_dist_lbl:
            correct_map[k] = lowest_dist_lbl
    if len(correct_map) > 0:
        print("correction map:")
        for k, v in correct_map.items():
            print(f"\t{k} -> {v}")
        print("\nCORRECTED Uniq Labels and stats:")
        for idx, v in enumerate(test_helper.INFER_PREDICTIONS):
            if v in correct_map:
                test_helper.INFER_PREDICTIONS[idx] = correct_map[v]
        corr_cnts = Counter(test_helper.INFER_PREDICTIONS)
        for k, v in corr_cnts.items():
            print(f"\tlabel text: [{k}], counts: {v}")

    else:
        print("** all tag labels are clean, so we don't need correction map. nice! **")

    int_lbls = [correct_pred_labels[x] for x in test_helper.INFER_LABELS]
    int_preds = [correct_pred_labels[x] for x in test_helper.INFER_PREDICTIONS]

    acc_metric = evaluate.load("accuracy")
    results = acc_metric.compute(references=int_lbls, predictions=int_preds)
    print(results)

