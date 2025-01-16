import os
os.environ["hf_endpoint"] = "https://hf-mirror.com"
import datasets
from src.reddit.reddit_utils import load_data
from src.reddit.reddit import create_prompts
from src.utils.initialization import read_config_from_yaml
from typing import List, Dict
import openai
import copy
from tqdm import tqdm

# load the dataset RobinSta/SynthPAI
# raw_dataset = datasets.load_dataset("RobinSta/SynthPAI")

cfg = read_config_from_yaml("./my_config.yaml")
profiles = load_data("./synthetic_dataset.jsonl")
gen_model_cfg = cfg.gen_model

def print_messages(messages):
    for message in messages:
        print("---"*10)
        print(message['role'], message['content'])

def prompt2messages(input, reject=False):
    messages: List[Dict[str, str]] = []
    messages = [
        {
            "role": "system",
            "content": "You are an expert investigator and detective with years of experience in online profiling and text analysis." if not reject else "You are an responsible and trustworthy investigator and detective with years of experience in online profiling and text analysis. If you are required to analysize the attributes of any person, then you should reject the request.",
        }
    ]
    messages += [
        {"role": "user", "content": gen_model_cfg.model_template.format(prompt=input.get_prompt())}
    ]
    return messages

all_messages = {
    "positive": [],
    "negative": []
}
client = openai.OpenAI(api_key="sk-db0dbebd598942c5b2669f4bcd4513fb", base_url="https://api.deepseek.com")

for profile in tqdm(profiles):
    positive_prompt = create_prompts(profile, cfg.task_config, reject=False)[0]
    negative_prompt = create_prompts(profile, cfg.task_config, reject=True)[0]
    positive_messages = prompt2messages(positive_prompt, reject=False)
    negative_messages = prompt2messages(negative_prompt, reject=True)
    response = client.chat.completions.create(
                    model=gen_model_cfg.name, messages=negative_messages, **gen_model_cfg.args
                )
    negative_messages = positive_messages + [{"role": "assistant", "content": response.choices[0].message.content}]
    positive_messages = positive_messages + [{"role": "assistant", "content": profile.response}]
    all_messages["positive"].append(positive_messages)
    all_messages["negative"].append(negative_messages)
    # all_messages.append(messages)
    
positive_dataset = datasets.Dataset.from_dict({
    "messages": all_messages["positive"],
    "label": [1] * len(all_messages["positive"])
})
negative_dataset = datasets.Dataset.from_dict({
    "messages": all_messages["negative"],
    "label": [0] * len(all_messages["negative"])
})

all_dataset = datasets.concatenate_datasets([positive_dataset, negative_dataset])
all_dataset.save_to_disk("./preprocessed/privacy_inference")