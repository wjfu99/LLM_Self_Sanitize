
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
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import bert_score
import evaluate
import os
import logging
from typing_extensions import Literal
from rich.logging import RichHandler
import collections
import math
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
random.seed(0)

def parse_max_memory(max_memory_list):
    """Parse max_memory argument from list of 'gpu_id:memory' strings to dict"""
    if max_memory_list is None:
        return None
    max_memory_dict = {}
    for item in max_memory_list:
        gpu_id, memory = item.split(':')
        max_memory_dict[int(gpu_id)] = memory
    return max_memory_dict

def get_logger(name: str, level: Literal["info", "warning", "debug"]) -> logging.Logger:
    rich_handler = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel[level.upper()])

    if not logger.handlers:
        logger.addHandler(rich_handler)

    logger.propagate = False

    return logger

logger = get_logger(__name__, "info")

def eval_rouge(generated, target, tokenizer=None, measure="fmeasure"):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, tokenizer=tokenizer)
    raw_scores = [scorer.score(t, r) for r, t in zip(generated, target)]
    avg_scores = {}
    for key in raw_scores[0].keys():
        avg_scores[key] = np.mean([getattr(score[key], measure) for score in raw_scores])
    return avg_scores, raw_scores
    # return scores

# HF instance use rouge f1 measure not the original recall
def eval_rouge_hf(generated, target, tokenizer=None):
    scorer = evaluate.load("rouge", tokenizer=tokenizer)
    scorer.add_batch(predictions=generated, references=target)
    scores = scorer.compute(rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scores

def eval_bleu(generated, target, tokenizer=None, max_order=4, smooth=False):
    if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        tokenizer = tokenizer.encode
    # if only one reference is provided make sure we still use list of lists
    if isinstance(target[0], str):
        target = [[ref] for ref in target]
    
    target = [[tokenizer(r) for r in ref] for ref in target]
    generated = [tokenizer(p) for p in generated]
    score = compute_bleu(
        reference_corpus=target, translation_corpus=generated, max_order=max_order, smooth=smooth
    )
    (bleu, precisions, bp, ratio, translation_length, reference_length) = score
    return {
        "bleu": bleu,
        "precisions": precisions,
        "brevity_penalty": bp,
        "length_ratio": ratio,
        "translation_length": translation_length,
        "reference_length": reference_length,
    }

def eval_bleu_hf(generated, target, tokenizer=None):
    scorer = evaluate.load("bleu")
    scorer.add_batch(predictions=generated, references=target)
    scores = scorer.compute(tokenizer=tokenizer)
    return scores

def eval_bert(generated, target):
    precision, recall, f1 = bert_score.score(generated, target, lang="en", verbose=True)
    precision = np.mean(precision.tolist())
    recall = np.mean(recall.tolist())
    f1 = np.mean(f1.tolist())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

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

class HierarchicalFFSelfMonitor(torch.nn.Module):
    def __init__(self, input_shape, output_shapes=[2, 3], dropout = 0.5):
        super().__init__()
        self.dropout = dropout
        
        self.bottle_neck =torch.nn.Sequential(
            torch.nn.Linear(input_shape, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            )
        self.linear_l1 = torch.nn.Linear(256, output_shapes[0])
        self.linear_l2 = torch.nn.Linear(256, output_shapes[1])
        self.output_l1 = torch.nn.Linear(output_shapes[0], output_shapes[0])
        self.output_l2 = torch.nn.Linear(output_shapes[0]+output_shapes[1], output_shapes[1])
        self.levels = len(output_shapes)
        
    def forward(self, x):
        x = self.bottle_neck(x)
        logits_l1 = self.output_l1(self.linear_l1(x))
        logits_l2 = self.output_l2(torch.cat((logits_l1, self.linear_l2(x)), dim=-1))
        return logits_l1, logits_l2
    
    def lloss(self, logits, labels):
        loss = 0
        for l in range(self.levels):
            loss += torch.nn.CrossEntropyLoss()(logits[l], labels[l])
        return loss
        
def prepare_model_info(model_name, layer_number: Union[int, str]=-1):
    coll_str = "[0-9]+" if layer_number==-1 else str(layer_number)
    model_info = {
        "tiiuae/falcon-40b" : {
            "ff": f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", 
            "att": f".*transformer.h.{coll_str}.self_attention.dense"
            },
        "tiiuae/falcon-7b" : {
            "ff": f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", 
            "att": f".*transformer.h.{coll_str}.self_attention.dense"
            },
        "tiiuae/falcon-7b-instruct" : {
            "ff": f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", 
            "att": f".*transformer.h.{coll_str}.self_attention.dense"
            },
        "meta-llama/Llama-2-13b-chat-hf": {
            "ff": f"model.layers.{coll_str}.mlp.up_proj", 
            "att": f"model.layers.{coll_str}.self_attn.o_proj"
            },
        "meta-llama/Llama-3.1-8B-Instruct": {
            "ff": f"model.layers.{coll_str}.mlp.up_proj", 
            "att": f"model.layers.{coll_str}.self_attn.o_proj"
            },
        "Qwen/Qwen2.5-72B-Instruct": {
            "ff": f"model.layers.{coll_str}.mlp.up_proj",
            "att": f"model.layers.{coll_str}.self_attn.o_proj"
            },
        "Qwen/Qwen2.5-32B-Instruct": {
            "ff": f"model.layers.{coll_str}.mlp.up_proj",
            "att": f"model.layers.{coll_str}.self_attn.o_proj"
            },
        "meta-llama/Meta-Llama-3-70B-Instruct": {
            "ff": f"model.layers.{coll_str}.mlp.up_proj",
            "att": f"model.layers.{coll_str}.self_attn.o_proj"
            },
        }
    return model_info[model_name]

def chat2str(messages):
    chat_text = ""
    for m in messages:
        chat_text += f"**************{m['role']}**************\n"
        chat_text += f"{m['content']}\n"
    return chat_text

def print_chat(messages):
    print(chat2str(messages))
    print("\n\n")

def save_ff_representation(mod, inp, out, ff_rep, layer_num):
    # Out size: (batch_size, seq_len, hidden_size)
    ff_rep[layer_num] = out.squeeze().detach().to(torch.float32).cpu().numpy()
    
def fetch_ff_representation(mod, inp, out, ff_rep, layer_name):
    # Out size: (batch_size, seq_len, hidden_size)
    ff_rep["current"] = out[0, -1, :].squeeze().detach()
    
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

def add_input_ids(example, tokenizer):
    example["input_ids"] = tokenizer.apply_chat_template(example["messages"], tokenize=True)
    return example

def add_res_start_idx(example, tokenizer):
    example["res_start_idx"] = len(tokenizer.apply_chat_template(example["messages"][:-1], tokenize=True))
    return example

def check_files_exist(*file_paths):
    """
    Check if the input file(s) exist at the given file path(s).

    Parameters:
        *file_paths (str): One or more strings representing the file path(s) to check.

    Returns:
        bool: True if all the files exist, False otherwise.
    """
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            return False
    return True


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Folder '{folder_path}' created.")
        
def RawJSONDecoder(index):
    class _RawJSONDecoder(json.JSONDecoder):
        end = None

        def decode(self, s, *_):
            data, self.__class__.end = self.raw_decode(s, index)
            return data
    return _RawJSONDecoder

def extract_json(s, index=0):
    while (index := s.find('{', index)) != -1:
        try:
            return json.loads(s, cls=(decoder := RawJSONDecoder(index)))
            index = decoder.end
        except json.JSONDecodeError:
            index += 1
    return None

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

def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)

# login("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")
# Model
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# train_dataset = CircuitBreakerDataset(tokenizer=tokenizer)