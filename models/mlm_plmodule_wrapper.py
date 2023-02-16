"""
    Naive HF T5 Model Pytorch Lightning Module Wrapper.

    Copyright (C) 2022~ ETRI Language Intelligence Research Section, Jong-hun Shin.
"""

import math
import pickle

import tqdm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
# pip install evaluate
import evaluate

from typing import Optional, Callable, Any, Union

from torch import nn

from transformers import (T5ForConditionalGeneration, get_linear_schedule_with_warmup,
                          AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM)

from transformers.optimization import (Adafactor, AdafactorSchedule)

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

from . import test_helper


class ETRIT5ConditionalGenModelLightningModule(pl.LightningModule):
    """
    Reusable, Tokenizer-independent, T5-like Model Wrapper for Pytorch-lightning.
      - data_collator implementation:
         * read HF dataset batch iterator
         * write HF-style encoded-batch output
    """
    def __init__(self, hf_config_path: str="",
                 model_or_path: str="",
                 tokenizer: Optional[Union[str, Callable]]=None,
                 data_collator: Optional[Callable[Any,Any]]=None,
                 optimizer: str="cpuadam",
                 learning_rate: float=1e-3, warmup_steps: int=0,
                 weight_decay: float=0.0, adam_epsilon: float=1e-8,
                 train_batch_size: int=256, val_batch_size: int=32,
                 **kwargs):
        super(ETRIT5ConditionalGenModelLightningModule, self).__init__()
        self.save_hyperparameters(ignore=['data_collator',])

        if hf_config_path != "":
            model_cfg = AutoConfig.from_pretrained(hf_config_path)
            #self.model = T5ForConditionalGeneration(model_cfg)
            self.model = AutoModelForSeq2SeqLM.from_config(model_cfg)
        elif model_or_path != "":
            #self.model = T5ForConditionalGeneration.from_pretrained(model_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_or_path)
        else:
            raise Exception("assign hf_config_path or model_or_path parameters to initialize model.")

        self.data_collator = data_collator
        self.acc_metric = evaluate.load("accuracy")
        self.tknizer = None

        if isinstance(tokenizer, str):
            self.tknizer = AutoTokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, Callable):
            self.tknizer = tokenizer

    def forward(self, **inputs):
        #print(type(inputs["input_ids"]))
        return self.model(**inputs)

    def configure_optimizers(self):
        """ Prepare optimizer and scheduler.

        AdamW와 Adafactor를 사용하여 학습 가능하며, AdamW를 사용하는 경우,
        Weight decay를 bias와 layernorm에는 적용하지 않도록 수정해야 한다. (huggingface transformers의 flax mlm 구현체에서 확인)

        scheduler는 linear warmup-linear decay로 구현되어 있었다.

        # FIXME: adafactor optimizer가 LM 학습에 더 도움이 되는지 확인하고, 구현 추가 필요함
        """
        # auto-wrapped model이 될 가능성을 위해, model을 다시 본다.
        model = self.trainer.model
        # CHECKME: no_decay target 이름이 맞는지 한번 더 확인할 것
        no_decay = ["bias", "LayerNorm.weight"]
        #full_params = [p for n, p in model.named_parameters()]
        optim_group_params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]

        # FIXME: dutch T5 학습에는 Adafactor + LR 5e-3을 선택. bfloat16을 쓸 수 있는 환경이라면 고려해보자.
        # 일단은 deepspeed FusedAdam (W_mode=true)을 선택적으로 쓸 수 있게.
        if self.hparams.optimizer == "cpuadam":
            #optimizer = FusedAdam(optim_group_params, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,)
            optimizer = DeepSpeedCPUAdam(optim_group_params, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,)
        elif self.hparams.optimizer == "adafactor":
            # 만약 optimizer에서 lr=None이 아니라 lr=0.001을 지정하는 경우라면
            # scale_parameter=False, relative_step=False로 지정 필요
            #optimizer = Adafactor(optim_group_params, scale_parameter=False,
            #                      clip_threshold=1.0, decay_rate=-0.8,
            #                      eps=(1e-30, 1e-3),
            #                      relative_step=False, warmup_init=False, lr=1e-3)
            optimizer = Adafactor(optim_group_params, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        else:
            optimizer = torch.optim.AdamW(optim_group_params, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,)

        # huggingface transformers의 NOAM scheduler 구현을 그대로 사용함.
        if self.hparams.optimizer == "adafactor":
            scheduler = AdafactorSchedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.data_collator is not None:
            batch_in = self.data_collator(batch)
            batch_in = batch_in.to(self.device)
        else:
            batch_in = batch

        if "token_type_ids" in batch_in:
            del batch_in["token_type_ids"]

        #print(batch_in["labels"].cpu().detach())

        s2s_lm_out = self(**batch_in)
        loss = s2s_lm_out.loss
        lm_logits = s2s_lm_out.logits

        n_words = lm_logits.shape[1]

        self.log("train_loss", loss.item(), on_step=True, prog_bar=False, sync_dist=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.data_collator is not None:
            batch_in = self.data_collator(batch)
            batch_in = batch_in.to(self.device)
        else:
            batch_in = batch

        labels = batch_in["labels"]
        if "token_type_ids" in batch_in:
            del batch_in["token_type_ids"]

        outputs = self(**batch_in)
        val_loss = outputs.loss
        logits = outputs.logits

        # 시작은 어차피 258.. 로 시작하므로.
        pred_seq = logits[0:].argmax(2)
        n_words = logits.shape[1]

        self.log("val_loss", val_loss.item(), on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=self.hparams.val_batch_size)

        return { "loss": val_loss, "preds": pred_seq, "labels": labels }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.data_collator is not None:
            batch_in = self.data_collator(batch)
            batch_in = batch_in.to(self.device)
        else:
            batch_in = batch

        labels = batch_in["labels"]
        #print(labels.cpu().detach().numpy())

        if "token_type_ids" in batch_in:
            del batch_in["token_type_ids"]

        outputs = self.model.generate(batch_in['input_ids'], do_sample=False, num_beams=1, max_length=256)
        #print(outputs.cpu().detach().numpy())
        return { "preds": outputs, "labels": labels }

    def test_epoch_end(self, output_results):
        labels = [x['labels'].cpu().detach().numpy() for x in output_results]
        preds = [x['preds'].cpu().detach().numpy() for x in output_results]
        test_helper.INFER_LABELS = labels
        test_helper.INFER_PREDICTIONS = preds

    def export_hf_model(self, model_save_path):
        """
        Export model as Huggingface Compatible Model (as T5ForConditionalGeneration)
        """
        # 모델 저장
        self.model.save_pretrained(model_save_path,
                                   is_main_process=True,
                                   push_to_hub=False,)
        # tokenizer 추가 정보 저장
        self.tknizer.save_pretrained(model_save_path)

    @classmethod
    def convert_deepspeed_checkpoint_to_fp32(cls, deepspeed_checkpoint_path, fp32_checkpoint_filepath):
        convert_zero_checkpoint_to_fp32_state_dict(deepspeed_checkpoint_path, fp32_checkpoint_filepath)
        print("\n\n** Given Deepspeed model converted to fp32 checkpoint successfully. **\n",
              "Now you can load Pytorch-lightning checkpoint with following python code:\n",
              f"model = {cls.__name__}.load_from_checkpoint('{fp32_checkpoint_filepath}', strict=False)")
        print("\nand You can safely ignore some missing state_dict, e.g. model.encoder.embed_tokens.weight.")

