# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""The KLUE benchmark"""

import json
import os
import re

import datasets

_KLUE_CITATION = """\
 @misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han
      and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh
      and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee
      and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang
      and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park
      and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_KLUE_DESCRIPTION = """\
The KLUE is introduced to make advances in Korean NLP.
Korean pre-trained language models (PLMs) have appeared to solve Korean NLP problems
since PLMs have brought significant performance gains in NLP problems in other languages.
Despite the proliferation of Korean language models, however, none of the proper evaluation datasets has been opened yet.
The lack of such benchmark dataset limits the fair comparison between the models and further progress on model architectures.
"""

_KLUE_RE_DESCRIPTION = """\
RE is a task to identify semantic relations between entity pairs in a text.
The relation is defined between an entity pair consisting of subject entity and object entity.
The goal is then to pick an appropriate relationship between these two entities.
"""

_KLUE_YNAT_DESCRIPTION = """\
TC is a task to predict a topic of a news headline.
The topic is one of 7 categories: politics, economy, society, culture, world, IT/science, and sports.
"""

_KLUE_NLI_DESCRIPTION = """\
NLI is a task to infer the relationship between a hypothesis sentence and a premise sentence.
Given the premise, the model determines if the hypothesis is true (entailment), false (contradiction), or undetermined (neutral).
"""

_KLUE_STS_DESCRIPTION = """\
STS is a task which aims to predict the semantic similarity of two input sentences as a real value between 0 and 5.
Note that we furthure binarized the prediction scores into two classes with a threshold score 3.0 (paraphrased or not) and evaluated with a classification metric.
"""

_KLUE_MRC_DESCRIPTION = """\
MRC is a task of evaluating model that can answer a question about a given text passage.
Specifically, we formulate the task as a span prediction task, where the answer is a text segment (coined as spans) in the passage.
"""

_KLUE_NER_DESCRIPTION = """\
NER is a task to detect the boundaries of named entities in unstructured text and to classify the types.
A named entity can be of one of predefined entity types such as person, location, organization, time expressions, quantities and monetary values.
"""

_KLUE_DP_DESCRIPTION = """\
DP is a task that aims at finding relational information among words.
The goal is to predict a graph structure and a dependency label of an input sentence based on the dependency grammar.
"""

_KLUE_WOS_DESCRIPTION = """\
DST is a task to predict slot and value pairs (dialogue states) from a task-oriented dialogue.
The potential pairs are predefined by a given task schema and knowledge base (KB).
"""

class KlueConfig(datasets.BuilderConfig):
    """BuilderConfig for KLUE"""

    def __init__(self, features, data_name, citation, url, label_classes=("False", "True"), **kwargs):
        """BuilderConfig for KLUE.
        Args:
            features: `list[string]`, list of the features that will appear in the
                feature dict. Should not include "label".
            data_url: `string`, url to download the zip file from.
            citation: `string`, citation for the data set.
            url: `string`, url for information about the data set.
            label_classes: `list[string]`, the list of classes for the label if the
                label is present as a string. Non-string labels will be cast to either
                'False' or 'True'.
            **kwargs: keyword arguments forwarded to super.
        """

        super(KlueConfig, self).__init__(version=datasets.Version("1.1.0"), **kwargs)
        self.features = features
        self.label_classes = label_classes
        self.data_name = data_name
        self.citation = citation
        self.url = url

class KLUE(datasets.GeneratorBasedBuilder):
    """The KLUE benchmark."""

    BUILDER_CONFIGS = [
        KlueConfig(
            name = "re",
            description=_KLUE_RE_DESCRIPTION,
            features=["sentence", "subject", "object", "label"],
            data_name="klue-re-v1.1",
            citation=_KLUE_CITATION,
            url="https://klue-benchmark.com/",
        ),
        KlueConfig(
            name="ynat",
            description=_KLUE_YNAT_DESCRIPTION,
            features=["title", "label"],
            data_name="ynat-v1.1",
            citation=_KLUE_CITATION,
            url="https://klue-benchmark.com/",
            label_classes=["정치", "경제", "사회", "생활문화", "세계", "IT과학", "스포츠"],
        ),
        KlueConfig(
            name="nli",
            description=_KLUE_NLI_DESCRIPTION,
            features=["premise", "hypothesis", "label"],
            data_name="klue-nli-v1.1",
            citation=_KLUE_CITATION,
            url="https://klue-benchmark.com/",
            label_classes=["entailment", "contradiction", "neutral"],
        ),
        KlueConfig(
            name="sts",
            description=_KLUE_STS_DESCRIPTION,
            features=["sentence1", "sentence2", "labels"],
            data_name="klue-sts-v1.1",
            citation=_KLUE_CITATION,
            url="https://klue-benchmark.com/",
        ),
        KlueConfig(
            name="mrc",
            description=_KLUE_MRC_DESCRIPTION,
            features=["title", "context", "question", "plausible_answer", "answers"],
            data_name="klue-mrc-v1.1",
            citation=_KLUE_CITATION,
            url="https://klue-benchmark.com/",
        ),
        KlueConfig(
            name="ner",
            description=_KLUE_NER_DESCRIPTION,
            features=["sentence", "labels"],
            data_name="klue-ner-v1.1",
            citation=_KLUE_CITATION,
            url="https://klue-benchmark.com/",
            label_classes=["O", "B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG", "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT"],
        ),
        KlueConfig(
            name="dp",
            description=_KLUE_DP_DESCRIPTION,
            features=["ids", "words", "lemma", "pos", "head", "relation"],
            data_name="klue-dp-v1.1",
            citation=_KLUE_CITATION,
            url="https://klue-benchmark.com/",
        ),
        KlueConfig(
            name="wos",
            description=_KLUE_WOS_DESCRIPTION,
            features=["domain", "dialogue", "prev_state", "label_state", "slot_names", "sys_text"],
            #features=["domain", "dialogue", "prev_state", "label_state", "sys_text"],
            data_name="wos-v1.1",
            citation=_KLUE_CITATION,
            url="https://klue-benchmark.com/",
        ),
    ]

    DEFAULT_CONFIG_NAME = "re"

    def _info(self):
        features = {feature: datasets.Value("string") for feature in self.config.features}
        if self.config.name == "re":
            features["subject"] = dict(
                {
                    "text": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                }
            )
            features["object"] = dict(
                {
                    "text": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                }
            )
            features["label"] = datasets.features.ClassLabel(names=_get_label_classes(self.config.name, self.config.data_name, "relation_list.json"))
        elif self.config.name in ["ynat", "nli"]:
            features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)
        elif self.config.name == "sts":
            features["labels"] = dict(
                {
                    "label": datasets.Value("float"),
                    "real-label": datasets.Value("float"),
                    "binary-label": datasets.Value("int32"),
                }
            )
            features["labels"]["binary-label"] = datasets.features.ClassLabel(names=[0,1])
        elif self.config.name == "mrc":
            features["plausible_answer"] = datasets.Value("bool")
            features["answers"] = datasets.features.Sequence(
                {
                    "text": datasets.Value("string"),
                    "start_idx": datasets.Value("int32"),
                }
            )
        elif self.config.name == "ner":
            features["sentence"] = datasets.features.Sequence(datasets.Value("string"))
            features["labels"] = datasets.features.Sequence(datasets.features.ClassLabel(names=self.config.label_classes))
            # jhshin added. 230807.
            features["tagged_sent"] = datasets.Value("string")
        elif self.config.name == "dp":
            features["ids"] = datasets.features.Sequence(datasets.Value("int32"))
            features["words"] = datasets.features.Sequence(datasets.Value("string"))
            features["lemma"] = datasets.features.Sequence(datasets.Value("string"))
            features["pos"] = datasets.features.Sequence(datasets.Value("string"))
            features["head"] = datasets.features.Sequence(datasets.Value("int32"))
            features["relation"] = datasets.features.Sequence(datasets.Value("string"))
        elif self.config.name == "wos":
            features["domain"] = datasets.features.Sequence(datasets.Value("string"))
            features["dialogue"] = datasets.features.Sequence(
                {
                    "role": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
            features["prev_state"] = datasets.features.Sequence(datasets.Value("string"))
            features["label_state"] = datasets.features.Sequence(datasets.Value("string"))
            #features["label_state_ids"] = datasets.features.Sequence(datasets.features.ClassLabel(names=_get_label_classes(self.config.name, self.config.data_name, "ontology.json"), id="-1"))
            features["slot_names"] = datasets.features.Sequence(datasets.features.ClassLabel(names=_get_label_classes(self.config.name, self.config.data_name, "ontology.json")))

        return datasets.DatasetInfo(
            description=_KLUE_DESCRIPTION + self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation,
        )

    def _split_generators(self, dl_manager):
        dl_dir = os.path.join(self.config.data_dir, self.config.data_name)
        print(dl_dir)
        if self.config.name in ["re", "ynat", "nli", "sts", "mrc", "wos"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(dl_dir, f"{self.config.data_name}_train.json"),
                        "split": datasets.Split.TRAIN,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(dl_dir, f"{self.config.data_name}_dev.json"),
                        "split":datasets.Split.TEST,
                    },
                ),
            ]
        elif self.config.name in ["ner", "dp"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(dl_dir, f"{self.config.data_name}_train.tsv"),
                        "split": datasets.Split.TRAIN,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(dl_dir, f"{self.config.data_name}_dev.tsv"),
                        "split": datasets.Split.TEST,
                    },
                ),
            ]

    def _generate_examples(self, data_file, split):
        print(f"data_filepath: {data_file}")
        with open(data_file, encoding="utf-8") as f:
            if self.config.name in ["re", "ynat", "nli", "sts", "mrc", "wos"]:
                json_lines = json.load(f)
            elif self.config.name in ["ner", "dp"]:
                lines = f.readlines()

            if self.config.name in ["re", "ynat", "nli", "sts"]:
                for idx, json_line in enumerate(json_lines, start=0):
                    if self.config.name == "re":
                        yield idx, {
                            "sentence":json_line["sentence"],
                            "subject": {"text":json_line["subject_entity"]["word"],
                                        "start":json_line["subject_entity"]["start_idx"],
                                        "end":json_line["subject_entity"]["end_idx"]},
                            "object": {"text": json_line["object_entity"]["word"],
                                        "start": json_line["object_entity"]["start_idx"],
                                        "end": json_line["object_entity"]["end_idx"]},
                            "label":json_line["label"],
                        }
                    elif self.config.name == "ynat":
                        yield idx, {
                            "title":json_line["title"],
                            "label":json_line["label"],
                        }
                    elif self.config.name == "nli":
                        yield idx, {
                            "premise":json_line["premise"],
                            "hypothesis":json_line["hypothesis"],
                            "label":json_line["gold_label"],
                        }
                    elif self.config.name == "sts":
                        yield idx, {
                            "sentence1":json_line["sentence1"],
                            "sentence2":json_line["sentence2"],
                            "labels":{"label":json_line["labels"]["label"],
                                      "real-label":json_line["labels"]["real-label"],
                                      "binary-label":json_line["labels"]["binary-label"]},
                        }
            elif self.config.name == "mrc":
                idx = 0
                for data in json_lines["data"]:
                    title = data["title"]
                    for paragraph in data["paragraphs"]:
                        context = paragraph["context"]
                        for qa in paragraph["qas"]:
                            answer_starts = []
                            answers = []
                            plausible = False
                            if qa["answers"]:
                                answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                                answers = [answer["text"] for answer in qa["answers"]]
                            else:
                                if "plausible_answers" in qa.keys():
                                    answer_starts = [answer["answer_start"] for answer in qa["plausible_answers"]]
                                    answers = [answer["text"] for answer in qa["plausible_answers"]]
                                    plausible = True

                            yield idx, {
                                "title": title,
                                "context": context,
                                "question": qa["question"],
                                "plausible_answer": plausible,
                                "answers": {
                                    "text": answers,
                                    "start_idx": answer_starts,
                                },
                            }
                            idx += 1
            elif self.config.name == "wos":
                idx = 0
                for data in json_lines:
                    domains = data["domains"]
                    d_history=[]
                    prev_state=[]
                    for dialogue in data["dialogue"]:
                        role = dialogue["role"]
                        text = dialogue["text"]
                        if role == "sys":
                            yield idx, {
                                "domain": domains,
                                "dialogue": d_history,
                                "prev_state": prev_state,
                                "label_state": state,
                                #"label_state_ids": state,
                                "slot_names": [],
                                "sys_text": text,
                            }
                            idx += 1
                            prev_state = state
                        elif role == "user":
                            state = dialogue["state"]
                        d_history.append({"role": role, "text": text})
            elif self.config.name in ["ner", "dp"]:
                filtered_lines = list(filter(lambda x: not x.startswith("##"), lines))
                ner_sent_lines = list(filter(lambda x: x.startswith("## klue-ner"), lines))

                size = len(filtered_lines)
                idx_list = [idx + 1 for idx, val in
                            enumerate(filtered_lines) if val == "\n"]

                sent_infos = [filtered_lines[i: j] for i, j in
                       zip([0] + idx_list, idx_list +
                           ([size] if idx_list[-1] != size else []))]

                def split_values(value):
                    split_list = [line.strip().split("\t") for line in value if not line == "\n"]
                    if self.config.name == "ner":
                        characters = [v[0] if len(v)==2 else "" for v in split_list]
                        bio = [v[1] if len(v)==2 else v[0] for v in split_list]
                        return [characters, bio] if len(characters) == len(bio) else [[],[]]
                    elif self.config.name == "dp":
                        ids = [v[0] for v in split_list if len(v)==6]
                        words = [v[1] for v in split_list if len(v)==6]
                        lemma = [v[2] for v in split_list if len(v)==6]
                        pos = [v[3] for v in split_list if len(v)==6]
                        head = [v[4] for v in split_list if len(v)==6]
                        deprel = [v[5] for v in split_list if len(v)==6]
                        ret_list = [ids, words, lemma, pos, head, deprel]
                        return ret_list if all(len(ret_list[0]) == len(l) for l in ret_list[1:]) else [[],[],[],[],[],[]]

                for idx, sent_info in enumerate(list(map(split_values, sent_infos)), start=0):
                    if self.config.name == "ner":
                        yield idx, {
                            "sentence": sent_info[0],
                            "labels": sent_info[1],
                            "tagged_sent": ner_sent_lines[idx][ner_sent_lines[idx].find("\t")+1:].strip(),
                        }
                    elif self.config.name == "dp":
                        yield idx, {
                            "ids": sent_info[0],
                            "words": sent_info[1],
                            "lemma": sent_info[2],
                            "pos": sent_info[3],
                            "head": sent_info[4],
                            "relation": sent_info[5],
                        }

def _get_label_classes(name, data_name, label_file):
    label_path = os.path.join(data_name, label_file)

    with open(label_path, encoding="utf-8") as f:
        json_lines = json.load(f)
        if name == "re":
            label_classes = json_lines["relations"]
        elif name == "wos":
            label_classes = []
            for key, values in json_lines.items():
                label_classes.append(f"{key}")
                #for value in values:
                #    label_classes.append(f"{key}-{value}")

    return label_classes





