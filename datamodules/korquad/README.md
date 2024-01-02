## HOW TO PREPARE DATASET

to use korquad v1.0, download datasets from https://korquad.github.io/KorQuad%201.0/:
  * training dataset: https://korquad.github.io/dataset/KorQuAD_v1.0_train.json
  * validation dataset: https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json

then copy files into current directory, then convert two json files to jsonlines format, by execute a script:
python3 ./convert_korquad_to_jsonl.py
