# CharFormer/GBST-combined T5 Implementation for Huggingface Transformers.

Copyright (C) 2023~, ETRI LIRS. Jong-hun Shin.

based on github.com/lucidrains/charformer-pytorch project for GBST implementation, which distributed under MIT License.

you can easily load GBSWT5 model with this code. import HOW-TO:
```
import GBSWT5

from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

# you can import model definition directly
from GBSWT5 import GBSWT5Config, GBSWT5ForConditionalGeneration, GBSWT5Model
```

model loading HOW-TOs:
```
cfg = AutoConfig.from_pretrained(config_json_filepath)
model = AutoModelForSeq2SeqLM.from_config(cfg)
```

```
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
```
