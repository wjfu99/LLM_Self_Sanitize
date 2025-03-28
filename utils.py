
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
import warnings
from typing import Union
random.seed(0)

# The class of self-monitor model
class FFSelfMonitor(torch.nn.Module):
    def __init__(self, input_shape, output_shape=4, dropout = 0.5):
        super().__init__()
        self.dropout = dropout
        
        self.linear_relu_stack =torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, output_shape)
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def prepare_model_info(model_name, layer_number: Union[int, str]=-1):
    coll_str = "[0-9]+" if layer_number==-1 else str(layer_number)
    model_info = {
        "falcon-40b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
        "falcon-7b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
        "falcon-7b-instruct" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
        "open_llama_13b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
        "open_llama_7b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
        "opt-6.7b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj"),
        "opt-30b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj", ),
        "Llama-2-13b-chat-hf": ("meta-llama", f"model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
        "Llama-3.1-8B-Instruct": ("meta-llama", f"model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
    }
    return model_info[model_name]

def save_fully_connected_hidden(mod, inp, out, hidden, layer_name):
    # Out size: (batch_size, seq_len, hidden_size)
    hidden["current"] = out[0, -1, :].squeeze().detach()
    
# def save_fully_connected_hidden(mod, inp, out, save_dict, layer_name):
#     save_dict[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


def save_attention_hidden(mod, inp, out, save_dict, layer_name):
    save_dict[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())

def data_preprocess(example, tokenizer, min_res_len=5, max_res_len=50):
    
    messages = example["messages"]
    # if len(messages) < 2: 
    #     continue
    truncated_message_index = len(messages) - 1
    # assert messages[truncated_message_index]["role"] == "assistant"
    messages_u = messages[:truncated_message_index]
    messages_a = messages[:truncated_message_index+1]
    u_ids = tokenizer.apply_chat_template(messages_u, tokenize=True)
    a_ids = tokenizer.apply_chat_template(messages_a, tokenize=True)
    assert a_ids[:len(u_ids)] == u_ids
    res_len = len(a_ids) - len(u_ids)
    all_len = len(a_ids)
    if res_len < min_res_len:
        warnings.warn("The response is too short, skip this example")
    example["input_ids"] = a_ids
    example["res_start_idx"] = len(u_ids)
    example["ori_str"] = tokenizer.decode(a_ids)
    # if len(preprocessed_dataset["input_ids"]) >= max_exa:
    #     break
    return example

def create_dataset( 
            tokenizer=None, 
            max_exa=4000,
            min_res_len=5,
            max_res_len=50,
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
    benign_dataset = {"input_ids": [], "res_start_idx": [], "qa_str": []}
    for example in ds:
        messages = example["messages"]
        if len(messages) < 2: 
            continue
        # truncated_message_index = random.randrange(1, len(messages), 2)
        truncated_message_index = 1
        assert messages[truncated_message_index]["role"] == "assistant"
        messages_u = messages[:truncated_message_index]
        messages_a = messages[:truncated_message_index+1]
        # last_len = len(messages[-1].split(" "))
        # truncated_token_index = random.randrange(1, , 2)
        
        u_ids = tokenizer.apply_chat_template(messages_u, tokenize=True)
        a_ids = tokenizer.apply_chat_template(messages_a, tokenize=True)
        assert a_ids[:len(u_ids)] == u_ids
        res_len = len(a_ids) - len(u_ids)
        all_len = len(a_ids)
        if res_len < min_res_len or all_len > 512:
            continue
        # truncated_token_index = len(u_ids) + random.randint(min_res_len, min(res_len, max_res_len))
        # input_ids = a_ids[:truncated_token_index]
        
        # formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")
        benign_dataset["input_ids"].append(a_ids)
        benign_dataset["res_start_idx"].append(len(u_ids))
        benign_dataset["qa_str"].append(tokenizer.decode(a_ids))
        if len(benign_dataset["input_ids"]) >= max_exa:
            break
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
    harm_ref_dataset = {"input_ids": [], "res_start_idx": [], "qa_str": []}
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
        if res_len < min_res_len:
            continue
        # truncated_token_index = len(u_ids) + random.randint(min_res_len, min(res_len, max_res_len))
        # input_ids = a_ids[:truncated_token_index]
        harm_ref_dataset["input_ids"].append(a_ids)
        harm_ref_dataset["res_start_idx"].append(len(u_ids))
        harm_ref_dataset["qa_str"].append(tokenizer.decode(a_ids))
        if len(harm_ref_dataset["input_ids"]) >= max_exa:
            break
    
    # ======================= Harmful Request--Response ======================= #
    messages = []
    harm_res_dataset = {"input_ids": [], "res_start_idx": [], "qa_str": []}
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
        if res_len < min_res_len:
            continue
        # truncated_token_index = len(u_ids) + random.randint(min_res_len, min(res_len, max_res_len))
        # input_ids = a_ids[:truncated_token_index]
        harm_res_dataset["input_ids"].append(a_ids)
        harm_res_dataset["res_start_idx"].append(len(u_ids))
        harm_res_dataset["qa_str"].append(tokenizer.decode(a_ids))
        if len(harm_res_dataset["input_ids"]) >= max_exa:
            break
        
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