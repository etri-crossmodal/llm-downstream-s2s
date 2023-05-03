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
from peft import (get_peft_config, get_peft_model,
                  LoraConfig, PrefixTuningConfig,
                  TaskType)

from models import test_helper


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
                 num_beams_for_test: int=1, max_predict_length: int=512,
                 tuning_method: str="finetune",
                 gradient_checkpointing: bool=False,
                 **kwargs):
        super(ETRIT5ConditionalGenModelLightningModule, self).__init__()
        self.save_hyperparameters(ignore=['data_collator',])
        self.peft_config = None

        if tuning_method not in ['lora', 'prefixtuning', 'finetune']:
            print(f"WARNING: tuning_method '{tuning_method}'. override to 'finetune' automatically.")
            tuning_method = 'finetune'
            self.hparams.tuning_method = 'finetune'

        if hf_config_path != "":
            model_cfg = AutoConfig.from_pretrained(hf_config_path, use_auth_token=True)
            #self.model = T5ForConditionalGeneration(model_cfg)
            self.model = AutoModelForSeq2SeqLM.from_config(model_cfg)
        elif model_or_path != "":
            #self.model = T5ForConditionalGeneration.from_pretrained(model_or_path, use_auth_token=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_or_path, use_auth_token=True)
        else:
            raise ValueError("assign hf_config_path or model_or_path parameters to initialize model.")

        if self.hparams.tuning_method == "lora":
            # LoRA: arXiv:2106.09685
            self.peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=16,
                target_modules=["q", "v"], lora_alpha=32, lora_dropout=0.01)
            self.model = get_peft_model(self.model, self.peft_config)
            print(f"* NOTICE: Parameter-Efficient Fine Tuning mode enabled to "
                  f"{self.hparams.tuning_method}.")
            self.model.print_trainable_parameters()
        elif self.hparams.tuning_method == 'prefixtuning':
            # P-Tuning v2: arXiv:2110.07602, successor of P-Tuning(Liu et al., 2021)
            # num_virtual_tokens must be changed with task setting.
            # see Appendix B. Prompt Length part of the paper.
            raise NotImplementedError("NOT WORKING: dimension mismatch when forward(). "
                                      "just use -tuning_method=lora.")
            self.peft_config = PrefixTuningConfig(peft_type="PREFIX_TUNING",
                                                  task_type=TaskType.SEQ_2_SEQ_LM,
                                                  inference_mode=False,
                                                  num_virtual_tokens=20,
                                                  token_dim=self.model.config.d_model,
                                                  num_transformer_submodules=1,
                                                  num_attention_heads=self.model.config.num_heads,
                                                  num_layers=self.model.config.num_layers,
                                                  encoder_hidden_size=self.model.config.d_model,
                                                  prefix_projection=True,)
            self.model = get_peft_model(self.model, self.peft_config)
            print(f"* NOTICE: Parameter-Efficient Fine Tuning mode enabled to "
                  f"{self.hparams.tuning_method}.")
            self.model.print_trainable_parameters()

        self.data_collator = data_collator
        self.acc_metric = evaluate.load("accuracy")
        self.tknizer = None

        if self.hparams.tuning_method == "finetune" and gradient_checkpointing is True:
            print("** Gradient Checkpointing Enabled, and computation cache will be disabled.")
            self.model.gradient_checkpointing_enable()

        if isinstance(tokenizer, str):
            self.tknizer = AutoTokenizer.from_pretrained(tokenizer, use_auth_token=True)
        elif isinstance(tokenizer, Callable):
            self.tknizer = tokenizer

    def forward(self, **inputs):
        #print(type(inputs["input_ids"]))
        return self.model(**inputs)

    def configure_optimizers(self):
        """ Prepare optimizer and scheduler.

        AdamW와 Adafactor를 사용하여 학습 가능하며, AdamW를 사용하는 경우,
        Weight decay를 bias와 layernorm에는 적용하지 않도록 수정해야 한다.
        (huggingface transformers의 flax mlm 구현체에서 확인)

        scheduler는 linear warmup-linear decay로 구현되어 있었다.
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

        # NOTICE: dutch T5 학습에는 Adafactor + LR 5e-3을 선택.
        # 일단은 deepspeed FusedAdam (W_mode=true)을 선택적으로 쓸 수 있게.
        if self.hparams.optimizer == "cpuadam":
            #optimizer = FusedAdam(optim_group_params,
            #                      lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,)
            optimizer = DeepSpeedCPUAdam(optim_group_params,
                                         lr=self.hparams.learning_rate,
                                         eps=self.hparams.adam_epsilon,)
        elif self.hparams.optimizer == "adafactor":
            # 만약 optimizer에서 lr=None이 아니라 lr=0.001을 지정하는 경우라면
            # scale_parameter=False, relative_step=False로 지정 필요
            optimizer = Adafactor(optim_group_params, scale_parameter=False,
                                  clip_threshold=1.0, decay_rate=-0.8,
                                  eps=(1e-30, 1e-3),
                                  relative_step=False, warmup_init=False, lr=self.hparams.learning_rate)
            #optimizer = Adafactor(optim_group_params, scale_parameter=True,
            #                      relative_step=True, warmup_init=True, lr=None)
        else:
            optimizer = torch.optim.AdamW(optim_group_params,
                                          lr=self.hparams.learning_rate,
                                          eps=self.hparams.adam_epsilon,)

        # huggingface transformers의 NOAM scheduler 구현을 그대로 사용함.
        if self.hparams.optimizer == "adafactor":
            #scheduler = AdafactorSchedule(optimizer)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
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
        val_acc = torch.eq(pred_seq, labels).sum() / labels.nelement()

        self.log("val_loss", val_loss.item(), on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=self.hparams.val_batch_size)
        self.log("val_acc", val_acc.item(), on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=self.hparams.val_batch_size)

        return { "loss": val_loss, "preds": pred_seq, "labels": labels }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.data_collator is not None:
            batch_in = self.data_collator(batch)
            batch_in = batch_in.to(self.device)
        else:
            batch_in = batch

        if "labels" in batch_in:
            labels = batch_in["labels"]
        else:
            labels = None
        #print(labels.cpu().detach().numpy())

        if "token_type_ids" in batch_in:
            del batch_in["token_type_ids"]

        outputs = self.model.generate(input_ids=batch_in['input_ids'],
                                      do_sample=False,
                                      num_beams=self.hparams.num_beams_for_test,
                                      max_new_tokens=self.hparams.max_predict_length,
                                      #max_time=5.0,
                                      )
        #print(outputs.cpu().detach().numpy())
        return { "preds": outputs, "labels": labels }

    def test_epoch_end(self, output_results):
        if output_results[0]['labels'] is not None:
            labels = [x['labels'].cpu().detach().numpy() for x in output_results]
        else:
            labels = None
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
        # tokenizer 추가 정보 저장은 PEFT 미 사용시에만 적용?
        self.tknizer.save_pretrained(model_save_path)

    @classmethod
    def convert_deepspeed_checkpoint_to_fp32(cls, deepspeed_checkpoint_path, fp32_checkpoint_filepath):
        convert_zero_checkpoint_to_fp32_state_dict(deepspeed_checkpoint_path, fp32_checkpoint_filepath)
        print("\n\n** Given Deepspeed model converted to fp32 checkpoint successfully. **\n",
              "Now you can load Pytorch-lightning checkpoint with following python code:\n",
              f"model = {cls.__name__}.load_from_checkpoint('{fp32_checkpoint_filepath}', strict=False)")
        print("\nand You can safely ignore some missing state_dict, "
              "e.g. model.encoder.embed_tokens.weight.")

