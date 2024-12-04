
from datasets import load_dataset, Dataset
import transformers
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
import json
import random
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
random.seed(0)

def create_dataset( 
            tokenizer=None, 
            num_examples=10000,
            ):

    one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"

    # ================ Model and Template Config  ================
    # Default configs
    sep_token = ""
    switch_select = [0]
    use_refusal_retain = False

    # ======================= Retain ======================= #
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    # rt_ds = ds.select_columns("messages")
    orig_s = []
    benign_dataset = {"input_ids": [], "res_start_idx": []}
    for example in ds:
        messages = example["messages"]
        if len(messages) < 2: 
            continue
        truncated_message_index = random.randrange(1, len(messages), 2)
        assert messages[truncated_message_index]["role"] == "assistant"
        messages_u = messages[:truncated_message_index]
        messages_a = messages[:truncated_message_index+1]
        # last_len = len(messages[-1].split(" "))
        # truncated_token_index = random.randrange(1, , 2)
        
        u_ids = tokenizer.apply_chat_template(messages_u, tokenize=True)
        a_ids = tokenizer.apply_chat_template(messages_a, tokenize=True)
        assert a_ids[:len(u_ids)] == u_ids
        res_len = len(a_ids) - len(u_ids)
        start_idx = 5
        if res_len < start_idx:
            continue
        truncated_token_index = len(u_ids) + random.randint(start_idx, res_len)
        input_ids = a_ids[:truncated_token_index]
        
        # formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")
        benign_dataset["input_ids"].append(input_ids)
        benign_dataset["res_start_idx"].append(len(u_ids))
        # orig_s.append(formatted_input)
    

    # ======================= Borderline Retain ======================= #
    # from https://github.com/paul-rottger/exaggerated-safety
    # TODO:This dataset have to be classified!!
    # with open(f'data/xstest_v2_completions_gpt4_gpteval.csv', newline='') as f:
    #     data = [dict(row) for row in csv.DictReader(f)]
    #     data = [row for row in data if row['final_label'] == "1_full_compliance"]
    # messages = []
    # for d in data:
    #     messages.append([{
    #         "role": "user",
    #         "content": d['prompt']
    #     }, {
    #         "role": "assistant",
    #         "content": d['completion']
    #     }])
    # brt_ds = datasets.Dataset.from_dict({"messages": messages})
    

    # ======================= Harmful Request--Refuse ======================= #
    with open("data/circuit_breakers_train.json") as file:
        dataset = json.load(file)

    random.shuffle(dataset)
    # dataset = dataset[:2000]
    refusal_retain_orig = []
    messages = []
    harm_ref_dataset = {"input_ids": [], "res_start_idx": []}
    for d in dataset:
        messages = [{
            "role": "user",
            "content": d['prompt']
        }, {
            "role": "assistant",
            "content": d['llama3_output']
        }]
        u_ids = tokenizer.apply_chat_template(messages[0:1], tokenize=True)
        a_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        assert a_ids[:len(u_ids)] == u_ids
        res_len = len(a_ids) - len(u_ids)
        start_idx = 5
        if res_len < start_idx:
            continue
        truncated_token_index = len(u_ids) + random.randint(start_idx, res_len)
        input_ids = a_ids[:truncated_token_index]
        harm_ref_dataset["input_ids"].append(input_ids)
        harm_ref_dataset["res_start_idx"].append(len(u_ids))
    
    # ======================= Harmful Request--Response ======================= #
    messages = []
    harm_res_dataset = {"input_ids": [], "res_start_idx": []}
    for d in dataset:
        messages = [{
            "role": "user",
            "content": d['prompt']
        }, {
            "role": "assistant",
            "content": d['output']
        }]
        u_ids = tokenizer.apply_chat_template(messages[0:1], tokenize=True)
        a_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        assert a_ids[:len(u_ids)] == u_ids
        res_len = len(a_ids) - len(u_ids)
        start_idx = 5
        if res_len < start_idx:
            continue
        truncated_token_index = len(u_ids) + random.randint(start_idx, res_len)
        input_ids = a_ids[:truncated_token_index]
        harm_res_dataset["input_ids"].append(input_ids)
        harm_res_dataset["res_start_idx"].append(len(u_ids))
        
    benign_dataset = Dataset.from_dict(benign_dataset).shuffle()
    harm_ref_dataset = Dataset.from_dict(harm_ref_dataset).shuffle()
    harm_res_dataset = Dataset.from_dict(harm_res_dataset).shuffle()
    return {
        "benign": benign_dataset,
        "harm_ref": harm_ref_dataset,
        "harm_res": harm_res_dataset
    }


# login("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")
# Model
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# train_dataset = CircuitBreakerDataset(tokenizer=tokenizer)