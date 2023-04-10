#!/usr/bin/env python3
"""
    Huggingface 호환 Seq2Seq 언어모델을 위한 추론 인터페이스.

    Copyright (C) 2023~, ETRI LIRS. Jong-hun Shin. all rights reserved.
"""
import os
import sys
import argparse
import logging

import tqdm
import torch

from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, utils
)


#utils.logging.set_verbosity_info()
logger = utils.logging.get_logger("")
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)


# Disable TF-TRT Warnings, we don't want to use tf2 for tensorboard.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_argparser():
    """ generate argument parser. """
    parser = argparse.ArgumentParser(description="T5-like, Seq2SeqLM inference tool with hf transformers.")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Huggingface Compatible, Base Model Path or Name. e.g. google/byt5-small")
    parser.add_argument("-a", "--adapter", type=str, default="",
                        help="Adapter Model Path.")
    parser.add_argument("-i", "--input", type=str,
                        help="input text file name. if it doesn't exist, get inputs from stdin.")
    parser.add_argument("-o", "--output", type=str,
                        help="output text filename. if it doesn't exist, print outputs to stdout.")
    parser.add_argument("-t", "--tokenizer", type=str,
                        help="if model path != tokenizer path, "
                        "you can use this argument to assign tokenizer.")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="maximum additional token/sequence length.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="if --input/-i doesn't exist, batch size will set to 1 automatically.")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="beam size.")

    return parser


def get_adaptered_model(basemodel_instance, adapter_path):
    try:
        from peft import PeftModel, PeftConfig

        # adapter_conf = PeftConfig.from_pretrained(adapter_path)
        if isinstance(basemodel_instance, str):
            basemodel_instance = AutoModelForSeq2SeqLM.from_pretrained(basemodel_instance, device_map="auto")
        logger.warning("Prepare PEFT Model from basemodel.")
        return PeftModel.from_pretrained(basemodel_instance, adapter_path)
    except ImportError as e:
        print("huggingface/peft module not found, install peft with 'pip install peft'")
        raise e

def get_linecounts(file_in):
    lines = 0
    buffer_size = 1024 * 1024
    rf = file_in.read
    buf = rf(buffer_size)
    while buf:
        lines += buf.count("\n")
        buf = rf(buffer_size)

    file_in.seek(0, 0)
    return lines


def do_generate(args, model_instance, tokenizer_instance, input_b):
    # FIXME: apply hf's text2text-generation pipeline for deepspeed-inference
    input_tk = tokenizer_instance(input_b,
                                  add_special_tokens=True,
                                  return_tensors="pt", padding=True,
                                  pad_to_multiple_of=8,)
    with torch.no_grad():
        input_tk.to('cuda')
        return model_instance.generate(input_ids=input_tk["input_ids"],
                                       max_new_tokens=args.max_seq_length,
                                       num_beams=args.beam_size)

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    progress_total = 0

    if args.adapter != "":
        model = get_adaptered_model(args.model, args.adapter)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, device_map="auto")

    if args.tokenizer is not None and args.tokenizer != "":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.input is not None and args.input != "":
        logger.warning("input file open: %s", args.input)
        in_f = open(args.input, "rt")
        progress_total = get_linecounts(in_f)
    else:
        logger.warning("-input option missing, get inputs from stdin.")
        in_f = sys.stdin
    if args.output is not None and args.output != "":
        logger.warning("output file open: %s", args.output)
        out_f = open(args.output, "wt")
    else:
        logger.warning("-output option missing, set output stream to stdout.")
        out_f = sys.stdout
    batch_size = args.batch_size if args.input is not None and args.input != "" else 1

    model.eval()

    inputs = []
    for aline in (in_f if progress_total == 0 else tqdm.tqdm(in_f, total=progress_total)):
        inputs.append(aline.strip())

        if len(inputs) >= batch_size:
            outputs = do_generate(args, model, tokenizer, inputs)
            output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True,)
            for output in output_strs:
                out_f.write(output + '\n')
            inputs = []

    if len(inputs) != 0:
        outputs = do_generate(args, model, tokenizer, inputs)
        output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True,)
        for output in output_strs:
            out_f.write(output + '\n')
        inputs = []

    in_f.close()
    out_f.close()
    logger.warning("done.")

