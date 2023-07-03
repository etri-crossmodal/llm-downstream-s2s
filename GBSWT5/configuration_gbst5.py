"""
   GBSWT5 model configuration.

   Copyright (C) 2023~ ETRI LIRS. Jong-hun Shin.
"""

from typing import Mapping
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxSeq2SeqConfigWithPast
from transformers.utils import logging


logger = logging.get_logger(__name__)
_BLOCKS = (
    (1, 0), (2, 0), (3, 0), (4, 0),
    (6, 0), (9, 0),
    #(12, 0), (12, 3), (12, 6), (12, 9)
)


class GBSWT5Config(PretrainedConfig):
    """ Based on models.t5. configuration_t5. T5Config in hf Transformers. """
    model_type = "gbswt5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model",
                     "num_attention_heads": "num_heads",
                     "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=384,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        max_subword_block_size=None,            # GBSWT-related options here from
        subword_blocks=_BLOCKS,
        downsample_factor=1,
        score_consensus_attn=True,
        z_loss=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        # GBSWT-related configurations
        self.max_subword_block_size = max_subword_block_size
        self.subword_blocks = subword_blocks
        self.downsample_factor = downsample_factor
        self.score_consensus_attn = score_consensus_attn

        # z_loss for computational stability.
        # see https://github.com/tensorflow/mesh/blob \
        #         /fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        # (1) logits이 0으로 부터 너무 멀어지게 드리프팅 되지 않도록 하여, bf16에서 발생하는
        # round-off error를 방지하기 위함. (2) 로짓이 normalized log-probabilities가 되도록 제고한다.
        self.z_loss = z_loss

        if self.subword_blocks is not None and isinstance(self.subword_blocks, list):
            for idx, elem in enumerate(self.subword_blocks):
                self.subword_blocks[idx] = tuple(elem)
            self.subword_blocks = tuple(self.subword_blocks)

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class GBSWT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    """ just copy of T5OnnxConfig. """
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        if self.use_past:
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        return common_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13

