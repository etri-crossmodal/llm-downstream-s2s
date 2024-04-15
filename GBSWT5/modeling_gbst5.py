"""
    hf transformers-compatible GBST + T5 Model implementation.

    several methods are copying from huggingface/transformers/models/t5/modeling_t5.py
    as Implementation Standards for compatibility. (version 4.28.1)

    hf transformers' modeling_t5.py file is distributed under Apache 2.0 License.

    Copyright (C) 2023, ETRI LIRS, Jong-hun Shin.
"""
import copy

from typing import Optional, Union, Tuple

import torch

from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import add_start_docstrings, PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.t5.modeling_t5 import (
    T5LayerNorm, T5Stack, T5LayerFF,
    T5Model, T5PreTrainedModel, T5ForConditionalGeneration, T5EncoderModel,
    T5DenseActDense, T5DenseGatedActDense, T5Attention,
    T5_START_DOCSTRING
)
from einops import rearrange, repeat, einsum

from .configuration_gbst5 import GBSWT5Config
from .gbst import GBSWT


logger = logging.get_logger(__name__)


class GBSWT5GroupedQueryAttention(nn.Module):
    """
    모듈 이름은 GBSWT5GroupedQueryAttention 이지만, 보통의 ByT5에 사용되는 T5Attention과도
    호환된다.

    GQA 구현에는 https://github.com/fkodom/grouped-query-attention-pytorch (t5.py) 에서 대부분의 코드를 가져옴.
    이 클래스 정의 코드는 MIT License로 배포된다. Copyright (c) 2022. Frank Odom.
    또한 베이스코드의 상당수는 Huggingface Transformers 4.34.1의 T5Attention에서 가져옴.

    GQA를 위한 uptraining은 기존 학습 데이터의 10% 정도면 MHA에 준하는 것으로 추정.
    """
    def __init__(self, config: GBSWT5Config, has_relative_attention_bias=False):
        super().__init__()

        if config.num_heads % config.kv_heads != 0:
            # 나누기가 가능해야 하므로, base 일 경우 12의 절반인 6 또는 4
            # large일 경우 16의 제곱근인 4와 8, 2가 가능하다.
            raise ValueError(
                f"n_heads ({config.num_heads}) must be divisible by kv_heads ({config.kv_heads})"
            )

        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # TODO: Check if we need to store 'kv_heads' and 'inner_dim' as a properties
        self.kv_heads = config.kv_heads         # 새 하이퍼파라미터: query_groups 수로 설정되어야 함
        # NOTE: Relative attention bias typically only used in the first layer
        # of a `T5Stack` module.
        # self.kv_dim = self.kv_heads * self.key_value_proj_dim # 사용 안함. 아래와 연동.

        # Mesh TensorFlow initialization to avoid scaling before softmax
        # self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # self.k = nn.Linear(self.d_model, self.kv_dim, bias=False)
        # self.v = nn.Linear(self.d_model, self.kv_dim, bias=False)
        # self.o = nn.Linear(self.kv_dim, self.d_model, bias=False)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.pruned_heads = set()  # type: ignore
        self.gradient_checkpointing = False
        self._relative_position_bucket = T5Attention._relative_position_bucket


    def prune_heads(self, heads):
        """ Transformer 4.34.1에서 그대로 들고옴 """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """ Transformer 4.34.1에서 그대로 들고옴 """
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """ Transformer 4.34.1에서 그대로 들고옴 """
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(  # noqa: C901
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection. 원래는 seq_len을 -1로 설정하고 n_heads를 기준으로 view를 바꿈. """
            # NOTE: Changed from the original definition in T5Attention.
            sequence_length = states.shape[1]
            return states.view(
                batch_size, sequence_length, -1, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape. 역시 마찬가지로 seq_len을 미리 고정하고, inner_dim을 바꾸도록 함"""
            # NOTE: Changed from the original definition in T5Attention.
            sequence_length = states.shape[2]
            return (
                states.transpose(1, 2)
                .contiguous()
                .view(batch_size, sequence_length, -1)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states. 이 파트는
            기본 T5 구현과 동등하다. """
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states: (batch_size, n_heads, seq_length, dim_per_head)
        grouped_queries = shape(self.q(hidden_states))

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # # compute scores
        # 여기서 달라진다. 아래가 원래 코드.
        # scores = torch.matmul(
        #     query_states, key_states.transpose(3, 2)
        # )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        # kv_heads 하이퍼파라미터를 기준으로 query/key grouping
        grouped_queries = rearrange(
            grouped_queries, "b (g h) n d -> b g h n d", h=self.kv_heads
        )
        grouped_keys = rearrange(
            key_states, "b (g h) s d -> b g h s d", h=self.kv_heads
        ).mean(dim=1)
        #scores = einsum(grouped_queries, grouped_keys, "b g h n d, b h s d -> b h n s")
        scores = torch.einsum("bghnd,bhsd->bhns", grouped_queries, grouped_keys)

        # ---- 다시 동일함
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    # NOTE: This is different from the original in T5Attention!
                    # (1, self.n_heads, real_seq_length, key_length),
                    # 원래의 헤드 수 대신 kv_heads로 변경
                    (1, self.kv_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )
            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                # (batch_size, n_heads, seq_length, key_length)
                position_bias = position_bias + mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        # NOTE: This is different from the original in T5Attention!
        # -- 다시 달라짐: position_bias_masked를 바로 scores에 넣지 않고, kv_heads에 맞게 rearrange.
        grouped_position_bias = rearrange(
            position_bias_masked, "b (g h) n s -> b g h n s", h=self.kv_heads
        ).mean(dim=1)

        scores += grouped_position_bias
        # attn_weights: (batch_size, kv_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # NOTE: This is different from the original in T5Attention!
        # attn_output = unshape(torch.matmul(attn_weights, value_states))
        # -- 역시 value_states를 바로 넣는 대신 grouped로 바꿔서 전달
        grouped_values = rearrange(
            value_states, "b (g h) s d -> b g h s d", h=self.kv_heads
        ).mean(dim=1)
        attn_output = unshape(torch.matmul(attn_weights, grouped_values))
        # 그루핑되어 있었던 것을 다시 원래대로
        attn_output = repeat(
            attn_output, "b s d -> b s (g d)", g=(self.n_heads // self.kv_heads)
        )
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)  # type: ignore
        return outputs


class GBSWT5LayerSelfAttention(nn.Module):
    """
    hf transformers, 4.37.0dev0, T5LayerSelfAttention 클래스를 가져와서,
    T5Attention 파트를 변형.
    """
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        if isinstance(config.kv_heads, int):
            self.SelfAttention = GBSWT5GroupedQueryAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        else:
            self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class GBSWT5LayerCrossAttention(nn.Module):
    """
    hf transformers, 4.37.0dev0, T5LayerCrossAttention 클래스를 가져와서,
    T5Attention 파트를 변형.
    """
    def __init__(self, config):
        super().__init__()
        if isinstance(config.kv_heads, int):
            self.EncDecAttention = GBSWT5GroupedQueryAttention(config, has_relative_attention_bias=False)
        else:
            self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class GBSWT5Block(nn.Module):
    """
    hf transformers, 4.37.0dev0, T5Block 클래스를 가져와서, SelfAttention 모듈을 변경함.
    __init__() 제외하고는, 나머지는 동일함.
    """
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(GBSWT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(GBSWT5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        # hidden-states, present_key_value_states, (self-attention position bias),
        # (self-attention weights), (cross-attention position bias), (cross-attention weights)
        return outputs



class GBSWT5PreTrainedModel(PreTrainedModel):
    config_class = GBSWT5Config
    base_model_prefix = "GBSWT5"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GBSWT5Block"]
    _keep_in_fp32_modules = ["wo"]

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights. 대부분은 T5PreTrainedModel을 따른다. """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module,
            (GBSWT5Model, GBSWT5ForConditionalGeneration, GBSWT5EncoderModel,),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "qa_outputs"):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                module.qa_outputs.bias.data.zero_()
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention) or isinstance(module, GBSWT5GroupedQueryAttention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
        elif isinstance(module, GBSWT):
            module._init_weights(factor)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (GBSWT5Stack)):
            module.gradient_checkpointing = value
        elif isinstance(module, (GBSWT5GroupedQueryAttention)):
            module.gradient_checkpointing = value
        elif isinstance(module, (T5Stack)):
            module.gradient_checkpointing = value
        elif isinstance(module, (T5Attention)):
            module.gradient_checkpointing = value


GBSWT5PreTrainedModel._shift_right = T5PreTrainedModel._shift_right


class GBSWT5Stack(GBSWT5PreTrainedModel):
    """
    implement GBST-enabled T5Model, based on HF Transformers's T5Stack.
    GBST will not be used when config.is_decoder, for information leakage problem.
    """
    def __init__(self, config: GBSWT5Config, embed_tokens :nn.Embedding=None):
        super().__init__(config)

        if not config.is_decoder:
            # override embed_tokens, apply GBWST
            self.embed_tokens = GBSWT(embed_tokens=embed_tokens,
                                      max_block_size=config.max_subword_block_size,
                                      blocks=config.subword_blocks,
                                      downsample_factor=config.downsample_factor,
                                      score_consensus_attn=config.score_consensus_attn,
                                      use_bn=config.gbst_batchnorm,)
        else:
            self.embed_tokens = embed_tokens

        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [GBSWT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing, same as T5 Stack.
        self.post_init()
        # for Model Parallel
        self.model_parallel = False
        self.device_map = False
        self.gradient_checkpointing = False
        self.downsample_factor = config.downsample_factor

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                inputs_embeds=None,
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        """ GBST 파트를 제외하면, T5Stack.forward() 구현을 그대로 복제하였다. """
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            #print(f"old: {input_shape}")
            if not self.is_decoder:
                inputs_embeds, attention_mask = self.embed_tokens(input_ids, attention_mask)
            else:
                inputs_embeds = self.embed_tokens(input_ids)
            input_shape = inputs_embeds.size()[:-1]
            # for downsample_factor > 1
            #print(f"new: {input_shape}")

        batch_size, seq_length = input_shape
        #print(f"bs: {batch_size}, sl: {seq_length}")

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        #print(f"mask_seq_length: {mask_seq_length}")

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # transformers 4.3x 에서 gradient checkpointing refactoring을 했지만, 여기에는 반영하지 않는다.
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            retval = tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
            if not self.is_decoder:
                return retval, attention_mask
            return retval

        # must be return downsampled attention_mask
        retval = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
        # return updated attention_mask
        if not self.is_decoder:
            return retval, attention_mask
        return retval

    def get_input_embeddings(self):
        return self.embed_tokens.embeds

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens.embeds = new_embeddings

# FIXME: Will be deprecated in transformer v5.
GBSWT5Stack.parallelize = T5Stack.parallelize
GBSWT5Stack.deparallelize = T5Stack.deparallelize


class GBSWT5Model(GBSWT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.embeds.weight", "decoder_embed_tokens.embeds.weight"]

    def __init__(self, config: GBSWT5Config):
        """ override T5Model """
        # override some default missing parameters for pretrained ByT5 models (e.g. google/byt5-small)
        if not hasattr(config, 'max_subword_block_size'):
            config.max_subword_block_size = None
        if not hasattr(config, 'subword_blocks'):
            config.subword_blocks = ((1, 0), (2, 0), (3, 0), (6, 0), (9, 0),)
        if not hasattr(config, 'downsample_factor'):
            config.downsample_factor = 1
        if not hasattr(config, 'score_consensus_attn'):
            config.score_consensus_attn = True

        super().__init__(config)

        # naive T5와 같이 embedding은 공유함
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_cfg = copy.deepcopy(config)
        encoder_cfg.is_decoder = False
        encoder_cfg.use_cache = False
        encoder_cfg.is_encoder_decoder = False
        self.encoder = GBSWT5Stack(encoder_cfg, self.shared)

        # Embedding base를 공유하기는 하지만, decoder에는 GBSWT를
        # 적용하지 않아야 한다.
        decoder_cfg = copy.deepcopy(config)
        decoder_cfg.is_decoder = True
        decoder_cfg.is_encoder_decoder = False
        decoder_cfg.num_layers = config.num_decoder_layers
        self.decoder = GBSWT5Stack(decoder_cfg, self.shared)

        self.post_init()

        self.model_parallel = False
        self.device_map = None

        # FIXME: Grouped Query Attention이 사용되고, 두 하이퍼파라미터가 다 있으면
        # T5Attention Instance를 모두 GBSWT5GroupedQueryAttention로 바꿔야 한다.

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                decoder_inputs_embeds: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        """
        중요한 것은, downsampling이 된 경우 attention_mask가 변경되므로,
        이를 반영해주는 것이 필요하다. hf transformers 4.29.1에서 복제함
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs, attention_mask = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            # inference mode (e.g. .generate()) - must dewrap encoder output 'tuple'
            encoder_outputs, attention_mask = encoder_outputs
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


GBSWT5Model.parallelize = T5Model.parallelize
GBSWT5Model.deparallelize = T5Model.deparallelize
GBSWT5Model.get_input_embeddings = T5Model.get_input_embeddings
GBSWT5Model.set_input_embeddings = T5Model.set_input_embeddings
GBSWT5Model.get_encoder = T5Model.get_encoder
GBSWT5Model._prune_heads = T5Model._prune_heads


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class GBSWT5ForConditionalGeneration(GBSWT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.embeds.weight",
                          "decoder_embed_tokens.embeds.weight",
                          "lm_head.weight"]

    def __init__(self, config: GBSWT5Config):
        # override some default missing parameters for pretrained ByT5 models (e.g. google/byt5-small)
        if not hasattr(config, 'max_subword_block_size'):
            config.max_subword_block_size = None
        if not hasattr(config, 'subword_blocks'):
            config.subword_blocks = ((1, 0), (2, 0), (3, 0), (6, 0), (9, 0),)
        if not hasattr(config, 'downsample_factor'):
            config.downsample_factor = 1
        if not hasattr(config, 'score_consensus_attn'):
            config.score_consensus_attn = True

        # Grandparent의 init를 그대로 상속, 나머지는 T5ForConditionalGeneration을 따름
        super().__init__(config)

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_cfg = copy.deepcopy(config)
        encoder_cfg.is_decoder = False
        encoder_cfg.use_cache = False
        encoder_cfg.is_encoder_decoder = False
        self.encoder = GBSWT5Stack(encoder_cfg, self.shared)

        # Embedding base를 공유하기는 하지만, decoder에는 GBSWT를
        # 적용하지 않아야 한다.
        decoder_cfg = copy.deepcopy(config)
        decoder_cfg.is_decoder = True
        decoder_cfg.is_encoder_decoder = False
        decoder_cfg.num_layers = config.num_decoder_layers
        self.decoder = GBSWT5Stack(decoder_cfg, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # FIXME: Grouped Query Attention이 사용되고, 두 하이퍼파라미터가 다 있으면
        # T5Attention Instance를 모두 GBSWT5GroupedQueryAttention로 바꿔야 한다.

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
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
                ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """
        중요한 것은 encoder outputs에서 수정된 attention_mask를 다시 반영해야 하는 것임
        downsampling이 들어간 경우, attention_mask가 변경되기 때문.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs, attention_mask = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            if isinstance(encoder_outputs, tuple) and len(encoder_outputs) == 2:
                # inference mode (e.g. .generate()) - must dewrap encoder output 'tuple'
                encoder_outputs, attention_mask = encoder_outputs

            if len(encoder_outputs) >= 4:
                # 먼저 dewrapping
                attention_mask = encoder_outputs[3]

            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # add z_loss for computational stability in bf16 amp.
            # see https://github.com/huggingface/transformers/pull/10956#issuecomment-820712267
            if self.config.z_loss != 0.0:
                log_z = lm_logits.view(-1).logsumexp(-1)
                loss += self.config.z_loss * log_z.square()

        if not return_dict:
            # 여기서 attention_mask를 반환시켜서, 업데이트를 하게 한다.
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs + attention_mask
            return ((loss,) + output) if loss is not None else output

        # FIXME: 여기서도 가능하면 attention_mask를 추가 payload로 얹어서 반환 필요함.
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


GBSWT5ForConditionalGeneration.parallelize = T5ForConditionalGeneration.parallelize
GBSWT5ForConditionalGeneration.deparallelize = T5ForConditionalGeneration.deparallelize
GBSWT5ForConditionalGeneration.get_input_embeddings = T5ForConditionalGeneration.get_input_embeddings
GBSWT5ForConditionalGeneration.set_input_embeddings = T5ForConditionalGeneration.set_input_embeddings
GBSWT5ForConditionalGeneration.get_output_embeddings = T5ForConditionalGeneration.get_output_embeddings
GBSWT5ForConditionalGeneration.set_output_embeddings = T5ForConditionalGeneration.set_output_embeddings
GBSWT5ForConditionalGeneration.get_encoder = T5ForConditionalGeneration.get_encoder
GBSWT5ForConditionalGeneration.prepare_inputs_for_generation = T5ForConditionalGeneration.prepare_inputs_for_generation
GBSWT5ForConditionalGeneration.prepare_decoder_input_ids_from_labels = T5ForConditionalGeneration.prepare_decoder_input_ids_from_labels
GBSWT5ForConditionalGeneration._reorder_cache = T5ForConditionalGeneration._reorder_cache
GBSWT5ForConditionalGeneration._prune_heads = T5Model._prune_heads


class GBSWT5EncoderModel(T5PreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.embeds.weight"]

    def __init__(self, config: GBSWT5Config):
        # override some default missing parameters for pretrained ByT5 models (e.g. google/byt5-small)
        if not hasattr(config, 'max_subword_block_size'):
            config.max_subword_block_size = None
        if not hasattr(config, 'subword_blocks'):
            config.subword_blocks = ((1, 0), (2, 0), (3, 0), (6, 0), (9, 0),)
        if not hasattr(config, 'downsample_factor'):
            config.downsample_factor = 1
        if not hasattr(config, 'score_consensus_attn'):
            config.score_consensus_attn = True

        super().__init__(config)

        # naive T5와 같이 embedding은 공유함
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_cfg = copy.deepcopy(config)
        encoder_cfg.is_decoder = False
        encoder_cfg.use_cache = False
        encoder_cfg.is_encoder_decoder = False
        self.encoder = GBSWT5Stack(encoder_cfg, self.shared)

        self.post_init()

        self.model_parallel = False
        self.device_map = None

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        downsampled 된 attention_mask를 함께 반환한다.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs, attention_mask = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs, attention_mask


GBSWT5EncoderModel.parallelize = T5EncoderModel.parallelize
GBSWT5EncoderModel.deparallelize = T5EncoderModel.deparallelize
GBSWT5EncoderModel.get_input_embeddings = T5EncoderModel.get_input_embeddings
GBSWT5EncoderModel.set_input_embeddings = T5EncoderModel.set_input_embeddings
GBSWT5EncoderModel.get_encoder = T5EncoderModel.get_encoder
GBSWT5EncoderModel._prune_heads = T5EncoderModel._prune_heads
