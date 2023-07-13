"""
    Huggingface T5를 위한 SequenceClassifcation Model 정의.
    BARTForSequenceClassification()을 참조하여 작성함.
    (https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bart/modeling_bart.py#L1460)

    Copyright (C) 2023~ ETRI LIRS, Jong-hun Shin.
"""

import copy

from typing import Optional, Union, Tuple

import torch

from transformers.models.t5.modeling_t5 import (
    T5Config, T5Stack, T5PreTrainedModel, T5Model,
    T5_INPUTS_DOCSTRING, T5_ENCODER_INPUTS_DOCSTRING,
    PARALLELIZE_DOCSTRING, DEPARALLELIZE_DOCSTRING,
)
from transformers.modeling_outputs import (
    TokenClassifierOutput, SequenceClassifierOutput,
    Seq2SeqSequenceClassifierOutput,
)

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import (
    assert_device_map, get_device_map
)


class T5ClassificationHead(torch.nn.Module):
    """ Head for sentence-level classification tasks. """
    def __init__(self, input_dim: int, inner_dim: int,
                 num_labels: int, pooler_dropout_prob: float,):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim, inner_dim)
        self.dropout = torch.nn.Dropout(p=pooler_dropout_prob)
        self.out_proj = torch.nn.Linear(inner_dim, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ dropout -> dense -> activation w/tanh -> dropout -> projection """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)

        return hidden_states


class T5EncoderForSequenceClassification(T5PreTrainedModel):
    """
    이 모듈은 Encoder 파트의 T5Stack만 사용함.
    Enc-Dec를 모두 사용하는 것은 T5ForSequenceClassification으로 별도 구현.

    e.g. BartForSequenceClassification 등은 Seq2SeqSequenceClassifierOutput ...?
    """
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight",
                                       "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)
        # self.model의 경우 T5EncoderModel을 불러도 되지만, T5PreTrainedModel의
        # 다른 구조와 동일하게 T5Stack을 직접 참조함.
        self.shared = torch.nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        if "num_labels" in kwargs:
            print(f"num_labels args found, override to {kwargs['num_labels']}.")
            config.num_labels = kwargs["num_labels"]
        if "problem_type" in kwargs:
            print(f"problem_type args found, override to {kwargs['problem_type']}.")
            config.problem_type = kwargs["problem_type"]

        if config.problem_type is None or config.problem_type == "":
            print(f"set to multi_label_classification")
            config.problem_type = "multi_label_classification"

        # 최종 output은 config.num_labels를 필요로 한다.
        # inner_dim과 input_dim이 같게 설정되어 있음.
        # FIXME: dropout_rate 대신, T5Config를 확장해서 BART 처럼
        # T5Config.classifier_dropout 인자를 따로 줄 수 있는게 좋음.
        print(f"num_labels: {config.num_labels}")
        print(f"problem_type: {config.problem_type}")
        self.classification_head = T5ClassificationHead(config.d_model,
                                                        config.d_model,
                                                        config.num_labels,
                                                        config.dropout_rate,)

        # self.model = T5EncoderModel로 정의했으면 아래와 같이 수동으로 초기화.
        #self.model._init_weights(self.classification_head.dense)
        #self.model._init_weights(self.classification_head.out_proj)

        # CHECK: classification_head 초기화가 잘 되는지?
        self.post_init()

        # model parallel
        self.model_parallel = False
        self.device_map = None


    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (get_device_map(len(self.encoder.block),
                                          range(torchl.cuda.device_count())
                                          ) if device_map is None else device_map)
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classification_head.to(self.encoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallize()
        self.encoder = self.encoder.to("cpu")
        self.classification_head = self.classification_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    # follows hf T5 implementations
    def get_input_embedddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeds):
        self.shared = new_embeds
        self.encoder.set_input_embeddings(new_embeds)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        same as HF T5 implementation: prunes heads of the model. see PreTrainedModel base class.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(processor_class="T5Tokenizer",
                                checkpoint="t5-small",
                                output_type=SequenceClassifierOutput,
                                config_class="T5Config",)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class="T5Config")
    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        r"""
        labels(`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               head_mask=head_mask, inputs_embeds=inputs_embeds,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict,)

        # use last hidden indices as sentence representation. position of <eos> tokens?

        # FIXME: encoder-only 모델에서는 이렇게 안해도 될 것 같은데?
        hidden_states = outputs[0]
        last_hidden_indices = ((input_ids != self.config.pad_token_id).sum(dim=-1)-1
                               ).unsqueeze(dim=-1).repeat(1, hidden_states.size(-1)).unsqueeze(1)
        sentence_repr = hidden_states.gather(dim=1, index=last_hidden_indices,).squeeze(1)
        logits = self.classification_head(sentence_repr)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or
                                                     labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_func = torch.nn.MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_func(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_func(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss_func(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_func = torch.nn.BCEWithLogitsLoss()
                loss = loss_func(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss = loss, logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


class T5ForSequenceClassification(T5PreTrainedModel):
    """
    실험적인 구현체.

    Enc-Dec를 모두 사용하는 BARTForSequenceClassification과 유사한데, decoder의 last_hidden_state만 사용하는
    BARTForSequenceClassification과 달리, classification layer에
    encoder_last_hidden_state + last_hidden_state (concat.)을 받아서 출력을 결정한다.
    이러한 디자인은 추후 encoder_hidden을 유지하고 decoder만 새로 돌려서 결과를 바꿔야 하는 경우에
    더 나은 결과를 볼 수 있는지 확인하기 위함. 특히 token-free 모델은 encoder가 무겁기 때문에,
    이와 같은 접근 방법이 유효할 수 있다.

    다른 아이디어로는, encoder로 base knowledge를 encode 한 상태에서, decoder에서 task-specific prefix를
    넣는 방법도 고민해 볼만하다. 먼저 적당한 문제가 있는지 부터 좀 찾고 맞춰서 구현해보자..
    """
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight",
                                       "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)

        # FIXME: 바꿔야 한다. T5Model을 직접 쓰지 말고, Encoder/Decoder block을 직접 정의하고
        # forward에 연결하도록 한다.
        self.model = T5Model(config)

        if "num_labels" in kwargs:
            print(f"num_labels args found, override to {kwargs['num_labels']}.")
            config.num_labels = kwargs["num_labels"]
        if "problem_type" in kwargs:
            print(f"problem_type args found, override to {kwargs['problem_type']}.")
            config.problem_type = kwargs["problem_type"]

        if config.problem_type is None or config.problem_type == "":
            print(f"set to multi_label_classification")
            config.problem_type = "multi_label_classification"


        # 최종 output은 config.num_labels를 필요로 한다.
        # inner_dim과 input_dim이 같게 설정되어 있음.
        # FIXME: dropout_rate 대신, T5Config를 확장해서 BART 처럼
        # T5Config.classifier_dropout 인자를 따로 줄 수 있는게 좋음.
        self.classification_head = T5ClassificationHead(config.d_model * 2,
                                                        config.d_model,
                                                        config.num_labels,
                                                        config.dropout_rate,)

        # self.model = T5EncoderModel로 정의했으면 아래와 같이 수동으로 초기화.
        #self.model._init_weights(self.classification_head.dense)
        #self.model._init_weights(self.classification_head.out_proj)

        self.post_init()

        # model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(processor_class="T5Tokenizer",
                                checkpoint="t5-small",
                                output_type=Seq2SeqSequenceClassifierOutput,
                                config_class="T5Config",)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class="T5Config")
    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.FloatTensor], Seq2SeqSequenceClassifierOutput]:
        r"""
        labels(`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Return:
        Union[Tuple[torch.FloatTensor], Seq2SeqSequenceClassifierOutput]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # FIXME: delete decoder_input_embeds
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             head_mask=head_mask, decoder_head_mask=decoder_head_mask,
                             cross_attn_head_mask=cross_attn_head_mask,
                             encoder_outputs=encoder_outputs,
                             inputs_embeds=inputs_embeds,
                             decoder_inputs_embeds=decoder_inputs_embeds,
                             use_cache=use_cache, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict,)

        # use last hidden indices as sentence representation. position of <eos> tokens?
        hidden_states = outputs.last_hidden_state
        enc_hidden_states = outputs.encoder_last_hidden_state
        # sentence_repr = hidden_states[:, 0, :] 과 같이 단순히 이를 그대로 사용하는 것은 부적절함
        sentence_repr = torch.cat([enc_hidden_states[:, 0, :], hidden_states[:, 0, :]], dim=1)
        logits = self.classification_head(sentence_repr)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or
                                                     labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_func = torch.nn.MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_func(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_func(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss_func(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_func = torch.nn.BCEWithLogitsLoss()
                loss = loss_func(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(loss = loss, logits=logits,
                                               past_key_values=outputs.past_key_values,
                                               decoder_hidden_states=outputs.decoder_hidden_states,
                                               decoder_attentions=outputs.decoder_attentions,
                                               cross_attentions=outputs.cross_attentions,
                                               encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                                               encoder_hidden_states=outputs.encoder_hidden_states,
                                               encoder_attentions=outputs.encoder_attentions)


