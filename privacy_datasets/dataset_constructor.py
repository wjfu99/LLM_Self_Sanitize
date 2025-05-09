import datasets
import functools
import argparse

def length_filter(example, length=2000):
    messages = example["messages"]
    m_len = 0
    for message in messages:
        m_len += len(message["content"].split())
    return m_len <= length

def add_label(example):
    example["label"] = 0
    return example

parser = argparse.ArgumentParser(description="Construct privacy datasets")
parser.add_argument("--datasets", type=str, nargs="+", default=["regular_chat", "system_prompt_clinical", "privacy_inference", "user_prompt"], help="List of datasets to construct")
parser.add_argument("--datasets_length", type=int, nargs="+", default=[1000, 1000, 1000, 1000], help="List of dataset lengths")
parser.add_argument("--use_cache", action="store_true", help="Whether to use cached datasets")
parser.add_argument("--test_size", type=float, default=0.3, help="Test size for train-test split")
parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the dataset")
parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split")
args = parser.parse_args()

# Load the datasets
dataset_list = dict(zip(args.datasets, args.datasets_length))

dataset_dict = {}
# preprocessing_function = functools.partial(data_preprocess, tokenizer=tokenizer)
for dataset in dataset_list:
    if dataset == "regular_chat":
        dataset_dict["regular_chat"] = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft").map(add_label)
    else:
        raw_dataset = datasets.load_from_disk(f"./privacy_datasets/preprocessed/{dataset}")
        dataset_dict[dataset] = raw_dataset

aio_dataset = {}
for key, dataset in dataset_dict.items():
    dataset = dataset.add_column("type", [key] * len(dataset))
    if key != "regular_chat":
        shard_length = dataset_list[key] // 2
        pos_dataset = dataset.filter(lambda x: x['label']==1).select(range(shard_length))
        neg_dataset = dataset.filter(lambda x: x['label']==0).select(range(shard_length))
        pos_datasets = pos_dataset.train_test_split(test_size=args.test_size, shuffle=args.shuffle, seed=args.seed)
        neg_datasets = neg_dataset.train_test_split(test_size=args.test_size, shuffle=args.shuffle, seed=args.seed)
        aio_dataset[key + "_train"] = datasets.concatenate_datasets([pos_datasets["train"], neg_datasets["train"]])
        aio_dataset[key + "_test"] = datasets.concatenate_datasets([pos_datasets['test'], neg_datasets['test']])
    else:
        dataset = dataset.filter(functools.partial(length_filter, length=2000))
        dataset = dataset.select(range(dataset_list[key])).train_test_split(test_size=args.test_size, shuffle=args.shuffle, seed=args.seed)
        aio_dataset[key + "_train"], aio_dataset[key + "_test"] = dataset["train"], dataset["test"]
        
        
aio_dataset = datasets.DatasetDict(aio_dataset)
aio_dataset.save_to_disk("./privacy_datasets/preprocessed/aio")