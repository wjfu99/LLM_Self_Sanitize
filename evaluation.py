import datasets
import argparse
from pathlib import Path
import utils
import re
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import wandb
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Evaluate the output of LLMs")
parser.add_argument("--output_dir", type=str, default="./results/response", help="The output directory for the dataset")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="The model name to use")
args = parser.parse_args()

logger = utils.get_logger(__name__, level="info")

# Using wandb for logging results
# run = wandb.init(
#     project="Self-Sanitize",
#     name=time.strftime("%m-%d_%H-%M-%S"),
#     config=args,
#     job_type="evaluate",
# )

logger.info("Loading the dataset...")
save_path = Path(args.output_dir)
results_paths = [p for p in save_path.iterdir() if p.is_dir()]

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

results = []
for results_path in results_paths:
    logger.info(f"\n==========================Evaluating {results_path.name}==========================")
    output_dataset = datasets.load_from_disk(results_path / args.model_name)
    for key, dataset in output_dataset.items():
        match = re.match(r"([a-zA-Z0-9_]+)_(train|test)", key)
        if match:
            dataset_name = match.group(1)
            split = match.group(2)
        else:
            continue
            raise ValueError(f"Invalid key format: {key}")
        if split == "train" or dataset_name != "system_prompt_clinical":
            continue
        
        pos_dataset = dataset.filter(lambda x: x['label']==1)  
        neg_dataset = dataset.filter(lambda x: x['label']==0)
        
        logger.info("=======Malicious Query Scores=======")
        rouge_scores, _ = utils.eval_rouge(pos_dataset["response"], pos_dataset["entities"], tokenizer, measure="recall")
        # rouge_scores_hf = utils.eval_rouge_hf(pos_dataset["response"], pos_dataset["entities"])
        bleu_scores = utils.eval_bleu(pos_dataset["response"], pos_dataset["entities"], tokenizer)
        bert_scores = utils.eval_bert(pos_dataset["response"], pos_dataset["entities"])
        logger.info(f"bleu: {bleu_scores["bleu"]}, rouge1: {rouge_scores['rouge1']}, rouge2: {rouge_scores['rouge2']}, rougel: {rouge_scores['rougeL']}, bert: {bert_scores['f1']}")
        results.append(["malicious", results_path.name, bleu_scores["bleu"], rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL'], bert_scores['f1']])
        
        logger.info("=======Regular Query Scores=======")
        ground_truth = [messages[-1]["content"] for messages in neg_dataset["messages"]]
        rouge_scores, _ = utils.eval_rouge(neg_dataset["response"], ground_truth, tokenizer, measure="fmeasure")
        bleu_scores = utils.eval_bleu(neg_dataset["response"], ground_truth, tokenizer)
        bert_scores = utils.eval_bert(neg_dataset["response"], ground_truth)
        logger.info(f"bleu: {bleu_scores["bleu"]}, rouge1: {rouge_scores['rouge1']}, rouge2: {rouge_scores['rouge2']}, rougel: {rouge_scores['rougeL']}, bert: {bert_scores['f1']}")
        results.append(["regular", results_path.name, bleu_scores["bleu"], rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL'], bert_scores['f1']])
    
results = pd.DataFrame(results, columns=["type", "defense", "bleu", "rouge1", "rouge2", "rougel", "bert"])
results = results.melt(id_vars=["type", "defense"], value_vars=["bleu", "rouge1", "rouge2", "rougel"], var_name="metric", value_name="score")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# 第一个子图：malicious
sns.barplot(
    data=results[results["type"] == "malicious"],
    x="defense", y="score", hue="metric", ax=axes[0]
)
axes[0].set_title("Malicious Samples")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

# 第二个子图：regular
sns.barplot(
    data=results[results["type"] == "regular"],
    x="defense", y="score", hue="metric", ax=axes[1]
)
axes[1].set_title("Regular Samples")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

# 统一图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Metric", loc="upper center", ncol=4)

# 美化布局并保存
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("results.png")