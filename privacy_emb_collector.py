import os
os.environ["hf_endpoint"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "/home/fuwenjie/luc0_data/hf-cache/main"

import functools
from datetime import datetime
from typing import Any, Dict
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
import datasets
from datasets import load_dataset
from collections import defaultdict, Counter
from functools import partial
import re
from string import Template
from dataset_constructor import create_dataset
from utils import data_preprocess
from huggingface_hub import login
import argparse
import utils
from copy import deepcopy

# Hf credentials
login("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")

parser = argparse.ArgumentParser(description="Collect privacy embeddings")
parser.add_argument("--datasets", type=str, nargs="+", default=["regular_chat", "system_prompt_clinical", "privacy_inference", "user_prompt"], help="List of datasets to construct")
parser.add_argument("--datasets_length", type=int, nargs="+", default=[1000, 1000, 1000, 1000], help="List of dataset lengths")
parser.add_argument("--aio_dataset", type=str, default="./privacy_datasets/preprocessed/aio", help="The dataset to use")
parser.add_argument("--output_dir", type=str, default="./results/embeddings", help="The output directory for the dataset")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The model name to use")
parser.add_argument("--layer_number", type=int, nargs="+", default=[32, 33, 34, 35, 36], help="The layer number to use")
parser.add_argument("--skip_res_tokens", type=int, default=0, help="The number of tokens to skip in the response")
parser.add_argument("--max_monitor_tokens", type=int, default=50, help="The maximum number of tokens to monitor")
parser.add_argument("--max_memory", type=str, nargs="+", default=None, help="Max memory per GPU in format 'gpu_id:memory' (e.g., '0:60GB')")
args = parser.parse_args()


model_info = utils.prepare_model_info(args.model_name, "(.*)")

def fetch_selected_tokens(ff_rep, token_pos):
    for layer_num in ff_rep.keys():
        ff_rep[layer_num] = ff_rep[layer_num][token_pos]
    return ff_rep

def compute_and_save_results():

    # Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "mistral" in args.model_name:
        tokenizer.chat_template = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"
    max_memory_dict = utils.parse_max_memory(args.max_memory)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                         torch_dtype=torch.bfloat16,
                                         device_map="balanced",
                                         max_memory=max_memory_dict,
                                         # load_in_4bit=True,
                                         trust_remote_code=True)
    
    # Load the datasets
    aio_dataset = datasets.load_from_disk(args.aio_dataset)
    aio_dataset = aio_dataset.map(functools.partial(utils.add_input_ids, tokenizer=tokenizer))
    aio_dataset = aio_dataset.map(functools.partial(utils.add_res_start_idx, tokenizer=tokenizer))

    # Prepare to save the internal states
    ff_rep = {}
    ff_hook = {}
    for name, module in model.named_modules():
        match = re.search(f'{model_info["ff"]}$', name)
        if match:
            layer_num = int(match.group(1))
            if layer_num in args.layer_number:
                ff_hook[layer_num] = module.register_forward_hook(partial(utils.save_ff_representation, ff_rep=ff_rep, layer_num=layer_num))

    # Generate results
    out_dataset = {}
    for key, dataset in aio_dataset.items():
        match = re.match(r"([a-zA-Z0-9_]+)_(train|test)", key)
        if match:
            dataset_name = match.group(1)
            split = match.group(2)
            # if split == "test": # Not skip test set
            #     continue
        else:
            raise ValueError(f"Invalid key format: {key}")
        
        results = defaultdict(list)
        for idx in tqdm(range(len(dataset))):
            entry = dataset[idx]
            with torch.inference_mode():
                ff_rep.clear()

                input_ids = torch.tensor([entry["input_ids"]]).to(model.device)
                qa_str = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                start_pos = entry["res_start_idx"] + args.skip_res_tokens
                length = len(input_ids[0])
                end_pos = min(length, start_pos+args.max_monitor_tokens)
                input_ids = input_ids[:, :end_pos]
                outputs = model(input_ids)
                ff_rep = fetch_selected_tokens(ff_rep, range(start_pos, end_pos))

                results['qa_str'].append(qa_str)
                results['start_pos'].append(start_pos)
                results['label'].append(entry['label'])
                results['ff_rep'].append(deepcopy(ff_rep))
        out_dataset[key] = results
    save_path = Path(f"{args.output_dir}/{args.model_name}/{datetime.now().month}_{datetime.now().day}.pickle")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as outfile:
        outfile.write(pickle.dumps(out_dataset))


if __name__ == '__main__':
    compute_and_save_results()