import os
os.environ["hf_endpoint"] = "https://hf-mirror.com"
import datasets
from src.reddit.reddit_utils import load_data
from src.reddit.reddit import create_prompts
from srcano.reddit.reddit import create_prompts as create_ano_prompts
from src.utils.initialization import read_config_from_yaml
from srcano.utils.initialization import read_config_from_yaml as read_ano_config_from_yaml
from typing import List, Dict
import openai
import copy
from tqdm import tqdm
import json
import srcano
from srcano.anonymized.anonymized import load_profiles

# load the dataset RobinSta/SynthPAI
# raw_dataset = datasets.load_dataset("RobinSta/SynthPAI")

cfg = read_config_from_yaml("raw_data/UAL/my_config.yaml")
profiles = load_data("raw_data/UAL/synthetic_dataset.jsonl")
cfg_ano = read_ano_config_from_yaml("./reddit_qwen25.yaml")
anonymized_profiles = load_profiles(cfg_ano.task_config)
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
    "negative": [],
    "anonymized": []
}
client = openai.OpenAI(api_key="sk-hrsgcptaqccbqhwiutusulvlctwmkqvinxktkwrarscddqjr", base_url="https://api.ap.siliconflow.com")

mapped_feature = {
    "income": "income_level",
    "age": "age",
    "gender": "sex",
    "location": "city_country",
    "pobp": "birth_city_country",
    "education": "education",
    "occupation": "occupation",
    "married": "relationship_status"
}

features = []
type_features = []
ground_truths = []

for profile, anonymized_profile in tqdm(zip(profiles, anonymized_profiles), total=len(profiles)):
    positive_prompt = create_prompts(profile, cfg.task_config, reject=False)[0]
    negative_prompt = create_prompts(profile, cfg.task_config, reject=True)[0]
    anonymized_prompt = create_ano_prompts(anonymized_profile, cfg_ano.task_config, idx=-1)[0]
    
    assert len(positive_prompt.gt) == 1
    
    positive_messages = prompt2messages(positive_prompt, reject=False)
    negative_messages = prompt2messages(negative_prompt, reject=True)
    anonymized_messages = prompt2messages(anonymized_prompt, reject=False)
    response = client.chat.completions.create(
                    model=gen_model_cfg.name, messages=negative_messages, **gen_model_cfg.args
                )
    negative_messages = positive_messages + [{"role": "assistant", "content": response.choices[0].message.content}]
    positive_messages = positive_messages + [{"role": "assistant", "content": profile.response}]
    all_messages["positive"].append(positive_messages)
    all_messages["negative"].append(negative_messages)
    all_messages["anonymized"].append(anonymized_messages)
    features.append(feature := positive_prompt.gt[0])
    type_features.append(mapped_feature[feature])
    ground_truths.append(str(profile.review_pii["synth"][feature]["estimate"]))
    
positive_dataset = datasets.Dataset.from_dict({
    "messages": all_messages["positive"],
    "anonymized_messages": all_messages["anonymized"],
    "label": [1] * len(all_messages["positive"]),
    "feature": features,
    "type_feature": type_features,
    "ground_truth": ground_truths
})
negative_dataset = datasets.Dataset.from_dict({
    "messages": all_messages["negative"],
    "anonymized_messages": all_messages["anonymized"],
    "label": [0] * len(all_messages["negative"]),
    "feature": features,
    "type_feature": type_features,
    "ground_truth": ground_truths
})

all_dataset = datasets.concatenate_datasets([positive_dataset, negative_dataset])
all_dataset.save_to_disk("./preprocessed/privacy_inference")