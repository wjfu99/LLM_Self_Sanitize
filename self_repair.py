import functools
from utils import data_preprocess
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM, Pipeline, DynamicCache

use_cache = True
model_name = "Llama-2-13b-chat-hf" #"opt-30b"
model_repos = "meta-llama"
dataset_length = {
    # "system_prompt": 1000,
    "regular_chat":1000,
    "system_prompt_clinical": 1000,
    "privacy_inference": 1000, 
    "user_prompt": 1000, 
}

# class SelfRepairPipeline(Pipeline):
#     def _sanitize_parameters(self, **kwargs):
#         preprocess_kwargs = {}
#         if "maybe_arg" in kwargs:
#             preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
#         return preprocess_kwargs, {}, {}
    
#     def preprocess(self, inputs, maybe_arg=2):
#         model_input = Tensor(inputs["input_ids"])
#         return {"model_input": model_input}
    
#     def _forward(self, model_inputs):
#         # model_inputs == {"model_input": model_input}
#         outputs = self.model(**model_inputs)
#         # Maybe {"logits": Tensor(...)}
#         return outputs
    
#     def postprocess(self, model_outputs):
#         best_class = model_outputs["logits"].softmax(-1)
#         return best_class


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(f'{model_repos}/{model_name}', device_map=device)
tokenizer = AutoTokenizer.from_pretrained(f'{model_repos}/{model_name}')

# Load the datasets
dataset_list = dataset_length.keys()
dataset_dict = {}
preprocessing_function = functools.partial(data_preprocess, tokenizer=tokenizer)
for dataset in dataset_list:
    if dataset == "regular_chat":
        dataset_dict["regular_chat"] = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft").map(preprocessing_function, load_from_cache_file=use_cache, num_proc=8)
    else:
        raw_dataset = datasets.load_from_disk(f"./privacy_datasets/preprocessed/{dataset}")
        preprocessed_dataset = raw_dataset.map(preprocessing_function, load_from_cache_file=use_cache, num_proc=8)
        dataset_dict[dataset] = preprocessed_dataset
for key, dataset in dataset_dict.items():
    if key != "regular_chat":
        shard_length = dataset_length[key] // 2
        pos_dataset = dataset.filter(lambda x: x['label']==1).select(range(shard_length))
        neg_dataset = dataset.filter(lambda x: x['label']==0).select(range(shard_length))
        dataset_dict[key] = datasets.concatenate_datasets([pos_dataset, neg_dataset])
    else:
        dataset = dataset.filter(lambda x: x['res_start_idx']<=2500)
        dataset_dict[key] = dataset.select(range(dataset_length[key]))
        

# TODO: Accelerate with larger batch size
for key, dataset in dataset_dict.items():
    for entry in dataset:
        with torch.inference_mode():
            messages = entry["messages"][:-1]
            assert messages[-1]["role"] == "user"
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
            unfinished_sequences = True
            past_key_values = DynamicCache()
            max_cache_length = past_key_values.get_max_cache_shape()
            original_input_ids = inputs["input_ids"].clone()
            input_length = original_input_ids.shape[1]
            while unfinished_sequences:
                outputs = model.generate(**inputs, past_key_values=past_key_values, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
                inputs["input_ids"] = outputs["sequences"]
                inputs["attention_mask"] = torch.ones_like(outputs["sequences"])
                if outputs["sequences"][0, -1] == tokenizer.eos_token_id:
                    break
                # completion = tokenizer.decode(outputs["sequences"][0, input_length: ], skip_special_tokens=True)
                # unfinished_sequences = stopping_criteria(input_ids, scores)
