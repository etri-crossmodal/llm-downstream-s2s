import os

# Disable TF-TRT Warnings, we don't want to use tf2 for tensorboard.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import multiprocessing as mp
import random
import functools
import argparse

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from datetime import date

import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import evaluate

from packaging.version import Version
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

import task_utils
from datamodules.klue_datasets import klue_eval_util


def get_argparser():
    """ generate argument parser. """
    parser = argparse.ArgumentParser(description="Train T5-like model with pytorch+transformers.")
    parser.add_argument("-model", type=str, required=True,
                        help="model(=a checkpoint of ETRIT5ConditionalGenModelLightningModule) "
                        "path, not huggingface-compatible model filepath.")
    parser.add_argument("-tokenizer", type=str, default="google/byt5-small",
                        help="set hf tokenizer name or model path, "
                        "because model doesn't have any tokenizer-related information.")

    parser.add_argument("-task", type=str, default="seq2seq",
                        help="set a downstream task. (seq2seq|nsmc-naive|nsmc-prompted|"
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

    parser.add_argument("-save_output", type=str, default="",
                        help="set path for saving predictions.")
    parser.add_argument("-save_label", type=str, default="",
                        help="set path for saving gold labels or target texts, if it exists.")

    parser.add_argument("-seed", type=int, default=123456,
                        help="set a seed for RNGs. if you assign value below 0(e.g. -1), "
                        "we will randomize seed with secrets.randbelow() function.")
    parser.add_argument("-batch_size", type=int, default=128,
                        help="train/valid data batch size")
    parser.add_argument("-gpus", type=int, default=1,
                        help="number of accelerators(e.g. GPUs) for training.")
    parser.add_argument("-float_precision", type=int, default=32,
                        help="set floating point precision. default value is 32, you can set 16. "
                        "if you set to 16 and bf16 supported, bf16 will be enabled automatically.")

    parser.add_argument("-beam_size", type=int, default=1,
                        help="beam size for testing/prediction step.")
    parser.add_argument("-max_predict_length", type=int, default=128,
                        help="maximum prediction string length.")

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

    accelerator_args = "gpu"
    accelerator_counts = args.gpus

    if args.gpus <= 0:
        accelerator_args = "cpu"
        accelerator_counts = int(len(os.sched_getaffinity(0)) // 1.5)

    # 기본적으로 Fused AdamW (DeepSpeed)는 Off, 32bit로 학습
    use_cpu_adam_arg = False
    precision_arg = args.float_precision
    callbacks = []

    if precision_arg != 32 and precision_arg != 16:
        raise Exception("bad argument: you can assign 32 or 16 for -float_precision")

    bf16_ready = (torch.version.cuda and torch.cuda.is_bf16_supported()
            and Version(torch.version.cuda) >= Version("11.0")
            and torch.distributed.is_nccl_available())

    if bf16_ready and precision_arg == 32:
        print("NOTICE: This CUDA GPU supports bfloat16 precision. "
              "We suggest you use '-float_precision 16' for faster inference.")
        #input("Press Enter to continue...")

    # we should use CPU adamW for deepspeed
    if precision_arg == 16 and bf16_ready:
        print("** bfloat16 available: enable bfloat16 training, instead of fp16.")
        precision_arg = "bf16"

    # ================ for retrieve task data ==================
    # 데이터 모듈, 이를 처리하기 위한 collator, 그리고 출력 label-id 를 mapping하는 dict를 받는다
    data_module, collator, label_id_map = task_utils.get_task_data(args.task,
                                                                   args.batch_size,
                                                                   args.tokenizer,
                                                                   args.train_data,
                                                                   args.valid_data,
                                                                   args.test_data,
                                                                   args.valid_data_proportions,
                                                                   args.test_data_proportions,
                                                                   args.max_seq_length,
                                                                   args.do_truncate)
    if data_module is None:
        raise Exception("invalid -task option argument.")
    # ==========================================================

    if Path(args.model).is_dir():
        try:
            interm_checkpoint_filename = Path(args.model).absolute().joinpath("./pytorch.ckpt")
            ETRIT5ConditionalGenModelLightningModule.convert_deepspeed_checkpoint_to_fp32(
                args.model, interm_checkpoint_filename)
        except:
            raise Exception("** DeepSpeed Stage 2/3 Checkpoint converting failed. maybe stage1?")
        model = ETRIT5ConditionalGenModelLightningModule.load_from_checkpoint(interm_checkpoint_filename,
                                                                              strict=False)
        os.unlink(interm_checkpoint_filename)
    else:
        model = ETRIT5ConditionalGenModelLightningModule.load_from_checkpoint(args.model, strict=False)

    # override hyperparameter for prediction
    model.hparams.num_beams_for_test = args.beam_size
    model.hparams.max_predict_length = args.max_predict_length

    model.tknizer = tknizer = AutoTokenizer.from_pretrained(args.tokenizer, use_auth_token=True)
    # set collator
    model.data_collator = collator

    # initialize trainer,
    # fsdp_native를 사용해야 할 경우, configure_optimizer에서
    # torch.optim.AdamW()에 self.trainer.model.parameters()가 전달되어야 함.
    # bf16을 쓰고 싶으면 ampere급 이상 GPU가 있는 곳에서 해라. 최소 A6000 필요함
    # 호스트 메모리가 적으면 nvme offloading도 고려해야 함
    # gradient accumulation을 사용하면 global_step이 그에 맞게 떨어진다.
    # learning rate scheduler를 위해서, max_epoch을 충분히 크게 잡을 것.

    trainer = pl.Trainer(accelerator=accelerator_args,
                         devices=accelerator_counts, num_nodes=1,
                         precision=precision_arg,
                         strategy="ddp",
            )
    trainer.test(model, datamodule=data_module)

    detokenized_preds = []
    def _decode_a_batch(grp):
        return tknizer.batch_decode(grp, skip_special_tokens=True,)

    # use half of available cores.
    effective_cpu_cnts = len(os.sched_getaffinity(0))//2

    # to remove useless detokenization process
    if args.task == 'klue-mrc':
        test_helper.INFER_LABELS = None

    # detokenize predicts and gold labels
    with mp.Pool(processes=effective_cpu_cnts) as pool:
        print(f"Detokenize Predicted Output and labels, with {effective_cpu_cnts} processes.")
        if test_helper.INFER_PREDICTIONS is not None:
            detokenized_preds = pool.map(_decode_a_batch, test_helper.INFER_PREDICTIONS)
            test_helper.INFER_PREDICTIONS = [item for sublist in detokenized_preds for item in sublist]
        if test_helper.INFER_LABELS is not None:
            detokenized_lbls = pool.map(_decode_a_batch, test_helper.INFER_LABELS)
            test_helper.INFER_LABELS = [item for sublist in detokenized_lbls for item in sublist]

    if args.task == 'klue-mrc':
        base_kluedata_dir = os.path.abspath(os.path.dirname(__file__))
        base_kluedata_dir += "/datamodules/klue_datasets/"
        mrcds = load_dataset(base_kluedata_dir + "/klue_data.py",
                             name="mrc", data_dir=base_kluedata_dir)
        # for testing purposes.
        #mrcds['test'] = mrcds['test'].shard(num_shards=100, index=99)

        test_helper.INFER_LABELS = []
        for idx, testdata in enumerate(mrcds['test']):
            newtd = {}
            newtd['id'] = str(idx)
            ans = testdata['answers']
            # rename klue mrc answer start_idx to answer_start
            ans = {'answer_start' if k == 'start_idx' else k:v for k, v in ans.items()}
            test_helper.INFER_LABELS.append({'id': str(idx), 'answers': ans})
        test_helper.INFER_PREDICTIONS = [{'prediction_text': apred, 'id': str(idx)}
                                         for idx, apred in enumerate(test_helper.INFER_PREDICTIONS)]

        # KLUE 평가 방법을 사용한 평가: EM/ROUGE-W
        em_scores, rouge_scores = [], []
        for idx, v in enumerate(test_helper.INFER_PREDICTIONS):
            pred_answer = v['prediction_text']
            pred_answer = klue_eval_util.normalize_answer_for_klue_mrc(pred_answer)
            ground_truths = [klue_eval_util.normalize_answer_for_klue_mrc(atruth)
                             for atruth in test_helper.INFER_LABELS[idx]['answers']['text']]
            #print(f"pred_answer: {pred_answer}")
            #print(f"ground truths: {str(ground_truths)}")

            em, rouge = klue_eval_util.compute_em_and_rouge_w_score_for_klue_mrc(pred_answer, ground_truths)
            em_scores.append(em)
            rouge_scores.append(rouge)

        print(f'(Official) KLUE MRC Eval - "exact_match": {np.mean(em_scores)}, "rouge": {np.mean(rouge_scores)}')

        # hf evaluate를 사용한 평가.
        squad_metric = evaluate.load("squad")
        squad_res = squad_metric.compute(references=test_helper.INFER_LABELS,
                                         predictions=test_helper.INFER_PREDICTIONS)
        print(f"SQuAD Metrics - {str(squad_res)}")

        # chrf, rouge는 references=[[str],], predictions=[str,] 을 받는다
        refs = [lbl['answers']['text'] for lbl in test_helper.INFER_LABELS]
        preds = [prd['prediction_text'] for prd in test_helper.INFER_PREDICTIONS]

        """
        # chrF는 backend인 sacrebleu의 요구사항대로, references 갯수가 모두 같아야 한다.
        # 그래서 제외됨.
        chrf_metric = evaluate.load("chrf")
        chrf_res = chrf_metric.compute(references=refs, predictions=preds)
        print(f"chrF - {str(chrf_res)}")
        """

        rouge_metric = evaluate.load("rouge")
        rouge_res = rouge_metric.compute(references=refs, predictions=preds)
        print(f"ROUGE Metrics - {str(rouge_res)}")
    else:
        print(f"# Test Labels: {len(test_helper.INFER_LABELS)}")
        print(f"# Test Predictions: {len(test_helper.INFER_PREDICTIONS)}")

        print(f"Predicted Unique labels(will include mis-typed label elements):")
        uniq_preds = task_utils.get_unique_labels(test_helper.INFER_PREDICTIONS)
        for k, v in uniq_preds.items():
            print(f"\tlabel text: [{k}], counts: {v}")

        if label_id_map is not None:
            print("\n* trying to correct mis-typed labels with levenshtein(edit) distance.")
            correction_map = task_utils.get_mislabel_correction_map(label_id_map, uniq_preds)

            if len(correction_map) > 0:
                print("correction map:")
                for k, v in correction_map.items():
                    print(f"\t{k} -> {v}")
                print("\nCORRECTED Uniq Labels and stats:")
                for idx, v in enumerate(test_helper.INFER_PREDICTIONS):
                    if v in correction_map:
                        test_helper.INFER_PREDICTIONS[idx] = correction_map[v]
                corr_cnts = Counter(test_helper.INFER_PREDICTIONS)
                for k, v in corr_cnts.items():
                    print(f"\tlabel text: [{k}], counts: {v}")
            else:
                print("** all tag labels are clean, so we don't need correction map. nice! **")

            # label text to label id mapping
            int_lbls = [label_id_map[x] for x in test_helper.INFER_LABELS]
            int_preds = [label_id_map[x] for x in test_helper.INFER_PREDICTIONS]

            # then calculate accuracy. FIXME: introduce f1 measure.
            acc_metric = evaluate.load("accuracy")
            results = acc_metric.compute(references=int_lbls, predictions=int_preds)
            print(results)
        else:
            print("\nWARNING: label-id map dictionary is None, so we evaluate EM/chrF/ROUGE. "
                  "or you can modify task_utils.py:get_task_data()")
            em_metric = evaluate.load("exact_match")
            em_res = em_metric.compute(references=test_helper.INFER_LABELS,
                                       predictions=test_helper.INFER_PREDICTIONS)
            print(f"Exact Match: {str(em_res)}")

            chrf_metric = evaluate.load("chrf")
            chrf_res = chrf_metric.compute(references=test_helper.INFER_LABELS,
                                           predictions=test_helper.INFER_PREDICTIONS)
            print(f"chrF: {str(chrf_res)}")

            rouge_metric = evaluate.load("rouge")
            rouge_res = rouge_metric.compute(references=test_helper.INFER_LABELS,
                                             predictions=test_helper.INFER_PREDICTIONS)
            print(f"ROUGE Metrics - {str(rouge_res)}")

    # save outputs and gold labels
    if args.save_output != "":
        with open(args.save_output, "wt") as out_f:
            for item in test_helper.INFER_PREDICTIONS:
                out_f.write(str(item) + '\n')
            out_f.close()

    if args.save_label != "":
        if test_helper.INFER_LABELS is not None:
            with open(args.save_label, "wt") as out_f:
                for item in test_helper.INFER_LABELS:
                    if isinstance(item, list):
                        out_f.write('\t'.join(item) + '\n')
                    else:
                        out_f.write(str(item) + '\n')
                out_f.close()
        else:
            print("ERROR: -save_label option is not empty, but gold label dataset not found.")

