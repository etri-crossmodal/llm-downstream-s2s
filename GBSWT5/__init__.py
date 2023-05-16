from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM

from .configuration_gbst5 import GBSWT5Config, GBSWT5OnnxConfig
from .modeling_gbst5 import GBSWT5Model, GBSWT5ForConditionalGeneration

AutoConfig.register("gbswt5", GBSWT5Config)
AutoModel.register(GBSWT5Config, GBSWT5Model)
AutoModelForSeq2SeqLM.register(GBSWT5Config, GBSWT5ForConditionalGeneration)

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
