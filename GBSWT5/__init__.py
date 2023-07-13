from transformers import (
    AutoConfig, AutoModel,
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    T5Config, MT5Config, BartModel,
)

from .configuration_gbst5 import GBSWT5Config, GBSWT5OnnxConfig
from .modeling_t5enc import T5EncoderForSequenceClassification
from .modeling_gbst5 import GBSWT5Model, GBSWT5ForConditionalGeneration, GBSWT5EncoderModel

AutoConfig.register("gbswt5", GBSWT5Config)
AutoModel.register(GBSWT5Config, GBSWT5Model)
AutoModelForSeq2SeqLM.register(GBSWT5Config, GBSWT5ForConditionalGeneration)

# T5EncoderForSequenceClassification 추가
AutoModelForSequenceClassification.register(T5Config, T5EncoderForSequenceClassification)
#AutoModelForSequenceClassification.register(T5Config, T5ForSequenceClassification)


def patch_sentence_transformers_models_Transformer():
    # compatible with sentence_transformers==2.2.2
    from sentence_transformers import models

    def _load_model(maybe_self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            maybe_self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, MT5Config):
            maybe_self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, GBSWT5Config):
            maybe_self._load_gbswt5_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            maybe_self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)
            print(f"Model type: {type(maybe_self.auto_model)}")

    def _load_mt5_model(maybe_self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the encoder model from T5"""
        from transformers import MT5EncoderModel
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        maybe_self.auto_model = MT5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)

    def _load_gbswt5_model(maybe_self, model_name_or_path, config, cache_dir, **model_args):
        GBSWT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        maybe_self.auto_model = GBSWT5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args)

    def _forward(maybe_self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features and not isinstance(maybe_self.auto_model, BartModel):
            trans_features['token_type_ids'] = features['token_type_ids']

        if isinstance(maybe_self.auto_model, GBSWT5EncoderModel):
            # need to update downsampled attention_mask
            output_states, attention_mask = maybe_self.auto_model(**trans_features,
                                                                  return_dict=False,
                                                                  return_resized_attention_mask=True)
            features.update({'attention_mask': attention_mask})
        else:
            output_states = maybe_self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if maybe_self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    models.Transformer._load_model = _load_model
    models.Transformer._load_mt5_model = _load_mt5_model
    models.Transformer._load_gbswt5_model = _load_gbswt5_model
    models.Transformer.forward = _forward

    return models.Transformer

"""
from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.modeling_auto import (
    MODEL_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    _LazyAutoMapping
)

CONFIG_MAPPING_NAMES["gbswt5"] = "GBSWT5Config"
MODEL_MAPPING_NAMES["gbswt5"] = "GBSWT5Model"
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES["gbswt5"] = "GBSWT5ForConditionalGeneration"

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)

MODEL_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES
)

_import_structure = {"configuration_gbst5": ["GBSWT5Config", "GBSWT5OnnxConfig"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_gbst5"] = [
        "GBSWT5ForConditionalGeneration",
        "GBSWT5Model",
    ]


if TYPE_CHECKING:
    from .configuration_gbst5 import GBSWT5Config, GBSWT5OnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gbst5 import (
            GBSWT5ForConditionalGeneration,
            GBSWT5Model,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
"""
