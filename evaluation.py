import datasets
import argparse
from pathlib import Path
import utils
import re
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Evaluate the output of LLMs")
parser.add_argument("--output_dir", type=str, default="./privacy_datasets/self_repair_results", help="The output directory for the dataset")

parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="The model name to use")
args = parser.parse_args()

logger = utils.get_logger(__name__, level="info")

logger.info("Loading the dataset...")
save_path = Path(f"{args.output_dir}/{args.model_name}")
output_dataset = datasets.load_from_disk(save_path)

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
    
    logger.info("==========================Malicious Query Scores==========================")
    rouge_scores, _ = utils.eval_rouge(pos_dataset["response"], pos_dataset["entities"])
    # rouge_scores_hf = utils.eval_rouge_hf(pos_dataset["response"], pos_dataset["entities"])
    bleu_scores = utils.eval_bleu(pos_dataset["response"], pos_dataset["entities"])
    logger.info(f"bleu: {bleu_scores["bleu"]}, rouge1: {rouge_scores['rouge1']}, rouge2: {rouge_scores['rouge2']}, rougel: {rouge_scores['rougeL']}")
    
    logger.info("==========================Regular Query Scores============================")
    ground_truth = [messages[-1]["content"] for messages in neg_dataset["messages"]]
    rouge_scores, _ = utils.eval_rouge(neg_dataset["response"], ground_truth)
    bleu_scores = utils.eval_bleu(neg_dataset["response"], ground_truth)
    logger.info(f"bleu: {bleu_scores["bleu"]}, rouge1: {rouge_scores['rouge1']}, rouge2: {rouge_scores['rouge2']}, rougel: {rouge_scores['rougeL']}")