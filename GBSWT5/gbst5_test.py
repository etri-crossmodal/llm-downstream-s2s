"""
    테스트 루틴. train_kebyt5.py가 있는 디렉터리에서 다음을 실행:
    $ python -m models.gbst_t5.gbst5_test
"""
from transformers import AutoConfig, AutoTokenizer
from ..gbst_t5.modeling_gbst5 import GBSWT5ForConditionalGeneration, GBSWT5Model
from ..gbst_t5.configuration_gbst5 import GBSWT5Config

tknizer = AutoTokenizer.from_pretrained("google/byt5-small")
model_cg = GBSWT5ForConditionalGeneration.from_pretrained("google/byt5-small")
model_base = GBSWT5Model.from_pretrained("google/byt5-small")

x = tknizer("<extra_id_0>, how are you?", return_tensors="pt")
y = tknizer("<extra_id_0> Hello <extra_id_1>", return_tensors="pt").input_ids
outputs = model_cg.forward(input_ids=x['input_ids'], attention_mask=x['attention_mask'], labels=y)

loss = outputs.loss
logits = outputs.logits
print(loss, logits.shape)

y2 = tknizer("<extra_id_0> ", return_tensors="pt").input_ids
decoder_input_ids = model_base._shift_right(y2)
outputs = model_base(input_ids=x["input_ids"],
                     decoder_input_ids=decoder_input_ids)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)
