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
import GBSWT5

from typing import Optional, Callable, Any, Union
from packaging.version import Version

from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from transformers import (T5ForConditionalGeneration,
                          get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,
                          AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM)
from transformers.optimization import (Adafactor, AdafactorSchedule)
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import grad_norm
from peft import (get_peft_config, get_peft_model,
                  LoraConfig, PrefixTuningConfig,
                  TaskType)

# DEPRECATED: no more use
from models import test_helper


"""
    copied from github:katsura-jp/pytorch-cosine-annealing-with-warmup,
    which distributed under MIT license.

    CosineAnnealingWarmupRestarts, Copyright (c) 2022 Naoki Katsura.
"""
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        self.last_epoch = last_epoch

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


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
                 lr_scheduler: str="cosanneal",     # LR Scheduler: one of ['cosanneal', 'linear']
                 learning_rate: float=1e-3, warmup_steps: int=0,
                 weight_decay: float=0.0, adam_epsilon: float=1e-7,     # adam_epsilon 1e-8 to 1e-7, for fp16 training.
                 train_batch_size: int=256, val_batch_size: int=32,
                 num_beams_for_test: int=1, max_predict_length: int=512,
                 tuning_method: str="finetune",
                 gradient_checkpointing: bool=False,
                 optim_cosanneal_gamma=0.75,        # Optimizer: Cosine-Annealing Gamma Hparam
                 optim_cosanneal_restarts=4,        # Optimizer: Cosine-Annealing Restarting number of times
                 optim_cosanneal_min_lr=1e-7,       # Optimizer: Cosine-Annealing Minimum LR rate
                 **kwargs):
        super(ETRIT5ConditionalGenModelLightningModule, self).__init__()
        self.save_hyperparameters(ignore=['data_collator',])
        self.peft_config = None

        if tuning_method not in ['lora', 'prefixtuning', 'finetune']:
            print(f"WARNING: tuning_method '{tuning_method}'. override to 'finetune' automatically.")
            tuning_method = 'finetune'
            self.hparams.tuning_method = 'finetune'

        if hf_config_path != "":
            model_cfg = AutoConfig.from_pretrained(hf_config_path, token=True)
            if isinstance(model_cfg, GBSWT5.GBSWT5Config):
                model_cfg.z_loss = 0.0
            #self.model = T5ForConditionalGeneration(model_cfg)
            self.model = AutoModelForSeq2SeqLM.from_config(model_cfg)
        elif model_or_path != "":
            #self.model = T5ForConditionalGeneration.from_pretrained(model_or_path, token=True)
            model_cfg = AutoConfig.from_pretrained(model_or_path, token=True)
            if isinstance(model_cfg, GBSWT5.GBSWT5Config):
                model_cfg.z_loss = 0.0      # 학습 성능을 위해 z_loss를 제외
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_or_path, config=model_cfg,
                                                               token=True)
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
        #self.acc_metric = evaluate.load("accuracy")
        self.tknizer = None

        if self.hparams.tuning_method == "finetune" and gradient_checkpointing is True:
            print("** Gradient Checkpointing Enabled, and computation cache will be disabled.")
            self.model.gradient_checkpointing_enable()

        if isinstance(tokenizer, str):
            self.tknizer = AutoTokenizer.from_pretrained(tokenizer, token=True)
        elif isinstance(tokenizer, Callable):
            self.tknizer = tokenizer

        self.model_cfg = model_cfg

    def forward(self, **inputs):
        #print(type(inputs["input_ids"]))
        return self.model(**inputs)

    def freeze_gbswt(self, freeze=True):
        if isinstance(self.model_cfg, GBSWT5.GBSWT5Config):
            # 만약 GBSWT 모델이면 GBST 레이어를 frozen.
            gbst_frozen_target = ['encoder.embed_tokens.embeds.weight',
                                  'encoder.embed_tokens.positional_convol.2.convol.weight',
                                  'encoder.embed_tokens.positional_convol.2.convol.bias',
                                  'encoder.embed_tokens.positional_convol.2.proj.weight',
                                  'encoder.embed_tokens.positional_convol.2.proj.bias',
                                  'encoder.embed_tokens.cand_scoring.0.weight',
                                  'encoder.embed_tokens.cand_scoring.0.bias',
                                  #'shared.weight',
                                  ]
            print("** GBST Model found, freeze GBSWT layers for training downstream.")
            for name, param in self.model.named_parameters():
                if name in gbst_frozen_target:
                    print(f"** freeze {name} layer.")
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            print("** GBST Layer not found, skip freezing GBST-related layers.")

    def freeze_shared_embeddings(self, freeze=True):
        whole = self.model
        print(f"** Freeze Shared embedding ({freeze})")
        whole.shared.weight.requires_grad = not freeze

    def freeze_encoder(self, freeze=True):
        """
        T5 구조 상, embeddings가 frozen 되면 model.shared.weight도 frozen,
        디코더의 embed_tokens.weight도 frozen 됨. encoder/decoder를 따로 분리 불가능
        """
        blacklist = []
        blacklist.append('embed_tokens.weight')

        print(f"** Freeze Encoder ({freeze})")
        enc = self.model.encoder
        for name, param in enc.named_parameters():
            if name not in blacklist:
                param.requires_grad = not freeze
            else:
                param.requires_grad = freeze

    def freeze_decoder(self, freeze=True):
        blacklist = []
        blacklist.append('embed_tokens.weight')
        print(f"** Freeze Decoder ({freeze})")

        dec = self.model.decoder
        for name, param in dec.named_parameters():
            if name not in blacklist:
                param.requires_grad = not freeze
            else:
                param.requires_grad = freeze

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
        no_decay = ["bias", "layer_norm.weight", "shared.weight", "embed_tokens",]
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
        elif self.hparams.optimizer == "adam8":
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW(optim_group_params,
                                           lr=self.hparams.learning_rate,
                                           betas=(0.9, 0.999),
                                           eps=self.hparams.adam_epsilon,
                                           optim_bits=8, min_8bit_size=16384,)
            except ImportError as e:
                print("we need bitsandbytes module, install bnb with 'pip install bitsandbytes' command.")
                raise(e)
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
        elif self.hparams.lr_scheduler == 'cosanneal':
            # Cyclic Cosine-Annealing Scheduler with Warm-up.
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=self.trainer.estimated_stepping_batches / self.hparams.optim_cosanneal_restarts,
                cycle_mult=1.0, max_lr=self.hparams.learning_rate,
                min_lr=self.hparams.optim_cosanneal_min_lr,
                warmup_steps=self.hparams.warmup_steps,
                gamma=self.hparams.optim_cosanneal_gamma,
                #num_training_steps=self.trainer.estimated_stepping_batches,
            )

        elif self.hparams.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            raise ValueError(f"undefined LR scheduler assigned: {self.hparams.lr_scheduler}")

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    # pytorch-lightning 2에서는 def on_before_optimizer_step(self, optimizer)로 구성.
    def _on_before_optimizer_step_v1(self, optimizer, optimizer_step):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def _on_before_optimizer_step_v2(self, optimizer, **kwargs):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

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

        # NOTE: lightning < 2 버전에서는 loss가 기본적으로 반환되기 때문에 중복되어 나타남
        self.log("loss", loss.item(), on_step=True, prog_bar=True, logger=True,
                 batch_size=self.hparams.train_batch_size,)

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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
        out_dict = { "preds": outputs.numpy(force=True),
                     "labels": labels.numpy(force=True) if labels is not None else None }
        return out_dict

    def export_hf_model(self, model_save_path):
        """
        Export model as Huggingface Compatible Model (as T5ForConditionalGeneration)
        """
        # 모델 저장
        # FIXME: import safetensors 및 safe_serialize=True 추가
        self.model.save_pretrained(model_save_path,
                                   is_main_process=True,
                                   safe_serialization=False,
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


# pytorch-lightning version에 따르는 on_before_optimizer_step() method.
ETRIT5ConditionalGenModelLightningModule.on_before_optimizer_step = \
    ETRIT5ConditionalGenModelLightningModule._on_before_optimizer_step_v1 if Version(pl.__version__) < Version("2.0") else \
    ETRIT5ConditionalGenModelLightningModule._on_before_optimizer_step_v2
