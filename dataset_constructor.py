from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
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

class CircuitBreakerDataset(Dataset):
    
    def __init__(self, 
                tokenizer=None, 
                num_examples=10000,
                ):
        super(CircuitBreakerDataset, self).__init__()

        self.max_length = 1024

        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"

        # ================ Model and Template Config  ================
        # Default configs
        sep_token = ""
        switch_select = [0]
        use_refusal_retain = False
        self.sep_token = sep_token

        # ======================= Retain ======================= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        # rt_ds = ds.select_columns("messages")
        orig_s = []
        benign_dataset = []
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
            benign_dataset.append(input_ids)
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
        harm_ref_dataset = []
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
            harm_ref_dataset.append(input_ids)
        
        # ======================= Harmful Request--Response ======================= #
        messages = []
        harm_res_dataset = []
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
            harm_res_dataset.append(input_ids)
        a = 1
    def __len__(self):
        return min(len(self.orig_s_retain), len(self.circuit_breaker_orig))
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        orig_s_retain = self.orig_s_retain[i]
        circuit_breaker_orig = self.circuit_breaker_orig[i]
        val_orig = self.val_orig[i % len(self.val_orig)]

        cb_tokenized_kwargs = dict(max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        tokenize_kwargs = dict(max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

        # =========== Circuit Breaker Inputs ===========
        # === split to [request, response] shape [512,512] to support different mask configs ===
        cb_request, cb_response = circuit_breaker_orig.split('<SEPARATOR>')
        self.tokenizer.padding_side = "left"
        tokenized_request_circuit_breaker = self.tokenizer(cb_request, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "right"
        response_tokenized_circuit_breaker = self.tokenizer(cb_response, add_special_tokens=False, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "left"

        combined_input_ids_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["input_ids"], response_tokenized_circuit_breaker["input_ids"]], dim=1)
        combined_attention_mask_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["attention_mask"], response_tokenized_circuit_breaker["attention_mask"]], dim=1)

        # ========== Retain Inputs ===========
        tokenized_inputs_retain = self.tokenizer(orig_s_retain.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)
        
        # =========== Val Inputs ===========
        tokenized_inputs_val = self.tokenizer(val_orig.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)

        return dict(
            input_ids_circuit_breaker=combined_input_ids_circuit_breaker,
            attention_mask_circuit_breaker=combined_attention_mask_circuit_breaker,
            input_ids=tokenized_inputs_retain["input_ids"],
            attention_mask=tokenized_inputs_retain["attention_mask"],
            input_ids_val=tokenized_inputs_val["input_ids"],
            attention_mask_val=tokenized_inputs_val["attention_mask"],
        )


login("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")
# Model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
train_dataset = CircuitBreakerDataset(tokenizer=tokenizer)