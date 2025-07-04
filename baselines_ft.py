import datasets
import argparse
from pathlib import Path
import utils
import re
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import json
from copy import deepcopy

parser = argparse.ArgumentParser(description="Run baselines")
parser.add_argument("--aio_dataset", type=str, default="./privacy_datasets/preprocessed/aio", help="The dataset to use")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="The model name to use")
parser.add_argument("--output_dir", type=str, default="./results/response", help="The output directory for the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="The batch size to use")

args = parser.parse_args()

logger = utils.get_logger(__name__, level="info")




logger.info("Loading the model...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, 
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.load_adapter(f"results/ft-llms/{args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="balanced",
    do_sample=True,
    num_return_sequences=1,
    batch_size=args.batch_size,
    max_new_tokens=512,
    torch_dtype=torch.bfloat16
)
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id


logger.info("Loading the dataset...")
aio_dataset = datasets.load_from_disk(args.aio_dataset)
keys = list(aio_dataset.keys())
for key in keys:
    match = re.match(r"([a-zA-Z0-9_]+)_(train|test)", key)
    if match:
        dataset_name = match.group(1)
        split = match.group(2)
    else:
        raise ValueError(f"Invalid key format: {key}")
    if split == "train" or dataset_name == "regular_chat":
        del aio_dataset[key]

baseline_datasets = {}
baseline_datasets["safety-ft"] = aio_dataset
logger.info("Applying instruction defense...")

logger.info("Generating responses for each baseline...")
for dataset_name, dataset in baseline_datasets.items():
    logger.info(f"Generating responses for {dataset_name}...")
    del_types = []
    for type in dataset.keys():
        if "query" in dataset[type].column_names:
            accomplished_messages_list = []
            response_list = []
            batched_dataset = dataset[type].batch(batch_size=args.batch_size)
            for batch in tqdm(batched_dataset, desc="Generating"):
                out = pipe(batch["query"])
                messages = [item[0]["generated_text"] for item in out]
                generated_text = [item[0]["generated_text"][-1]["content"] for item in out]
                accomplished_messages_list.extend(messages)
                response_list.extend(generated_text)
            dataset[type] = dataset[type].add_column("response", response_list)
            dataset[type] = dataset[type].add_column("accomplished_messages", accomplished_messages_list)
        else:
            logger.info(f"Skipping {dataset_name} as this baseline is not applicable to {type} benchmark.")
            del_types.append(type)
    for type in del_types:
        del dataset[type]
    dataset.save_to_disk(f"{args.output_dir}/{dataset_name}/{args.model_name}")
