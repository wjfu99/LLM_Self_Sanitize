from collections import defaultdict
import pickle
from pathlib import Path
import numpy as np
import scipy as sp
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import random
from utils import FFSelfMonitor, HierarchicalFFSelfMonitor
import utils
from tqdm import tqdm
import argparse
import logging
import re
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Self-monitoring model")
parser.add_argument("--embeddings_dir", type=str, default="./privacy_datasets/embeddings", help="The directory to save the embeddings")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="The model name to use")
parser.add_argument("--class_num", type=int, default=4, help="Number of classes")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps for training")
parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps for evaluation")
parser.add_argument("--dropout_mlp", type=float, default=0.5, help="Dropout rate for MLP")
parser.add_argument("--dropout_gru", type=float, default=0.25, help="Dropout rate for GRU")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")
parser.add_argument("--layer_number", type=int, nargs="+", default=[32, 33, 34, 35, 36], help="The layer number to use")
parser.add_argument("--hierarchical", action="store_true", default=True, help="Whether to use hierarchical self-monitoring")

args = parser.parse_args()

gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

logger = utils.get_logger(__name__, level="info")

# inference_results = list(Path("./results/").rglob("*.pickle"))
# print (inference_results)
    
class RNNHallucinationClassifier(torch.nn.Module):
    def __init__(self, dropout=args.dropout_gru):
        super().__init__()
        hidden_dim = 128
        num_layers = 4
        self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, seq):
        gru_out, _ = self.lstm(seq)
        return self.linear(gru_out)[-1, -1, :]
    
    
def gen_classifier_roc(train_features, train_labels, test_features, test_labels):
    X_train, X_test, y_train, y_test = train_features, test_features, train_labels, test_labels
    
    classifier_model = FFSelfMonitor(X_train.shape[1]).to(device)
    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(torch.long).to(device)
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(torch.long).to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for _ in range(1001):
        optimizer.zero_grad()
        sample = torch.randperm(X_train.shape[0])[:args.batch_size]
        pred = classifier_model(X_train[sample])
        loss = F.cross_entropy(pred, y_train[sample])
        loss.backward()
        optimizer.step()
    classifier_model.eval()
    with torch.no_grad():
        pred = F.softmax(classifier_model(X_test), dim=1)
        prediction_classes = torch.argmax(pred, dim=1).cpu()
    save_path = Path(f"./self_monitor_models/binary/classifier_model_layer{layer}.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier_model.state_dict(), save_path)
    logger.info(f"Classifier model saved to {save_path}")
    prediction_classes = (pred[:,1]>0.5).type(torch.long).cpu()
    return roc_auc_score(y_test.cpu(), pred[:,1].cpu()), (prediction_classes.numpy()==y_test.cpu().numpy()).mean()

def hierachical_gen_classifier_roc(train_features, train_labels, train_l2_labels, 
                                   test_features, test_labels, test_l2_labels):
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = \
    train_features, test_features, train_labels, test_labels, train_l2_labels, test_l2_labels
    
    classifier_model = HierarchicalFFSelfMonitor(X_train.shape[1]).to(device)
    X_train = torch.tensor(X_train).to(device)
    y1_train = torch.tensor(y1_train).to(torch.long).to(device)
    y2_train = torch.tensor(y2_train).to(torch.long).to(device)
    X_test = torch.tensor(X_test).to(device)
    y1_test = torch.tensor(y1_test).to(torch.long).to(device)
    y2_test = torch.tensor(y2_test).to(torch.long).to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    def evaluate(y1, y2, pred_l1, pred_l2):
        mask = y2 != -100
        pred_l2 = pred_l2[mask]
        y2 = y2[mask]
        with torch.inference_mode():
            loss_l1 = F.cross_entropy(pred_l1, y1)
            loss_l2 = F.cross_entropy(pred_l2, y2)
            prob_l1 = F.softmax(pred_l1, dim=1)
            prob_l2 = F.softmax(pred_l2, dim=1)
            acc_l1 = (torch.argmax(prob_l1, dim=1).cpu().numpy() == y1.cpu().numpy()).mean()
            acc_l2 = (torch.argmax(prob_l2, dim=1).cpu().numpy() == y2.cpu().numpy()).mean()
            auc_l1 = roc_auc_score(y1.cpu(), prob_l1[:,1].cpu())
            auc_l2 = roc_auc_score(y2.cpu(), prob_l2.cpu(), multi_class='ovr')
        return (loss_l1, loss_l2), (acc_l1, acc_l2), (auc_l1, auc_l2)

    for step in range(args.max_steps):
        if step % args.eval_steps == 0:
            evaluate_loss, evaluate_accuracy, evaluate_auc = evaluate(y1_test, y2_test, *classifier_model(X_test))
            logger.info(f"Step {step}: Evaluate loss: {evaluate_loss}, Evaluate accuracy: {evaluate_accuracy}, Evaluate AUC: {evaluate_auc}")
        optimizer.zero_grad()
        sample = torch.randperm(X_train.shape[0])[:args.batch_size]
        pred_l1, pred_l2 = classifier_model(X_train[sample])
        loss = classifier_model.lloss([pred_l1, pred_l2], [y1_train[sample], y2_train[sample]])
        loss.backward()
        optimizer.step()
    save_path = Path(f"./self_monitor_models/hierarchical/classifier_model_layer{layer}.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier_model.state_dict(), save_path)
    logger.info(f"Classifier model saved to {save_path}")
    return evaluate_accuracy, evaluate_auc
        
logger.info("Loading results from pickle file")
results_dir = Path(f"{args.embeddings_dir}/{args.model_name}")
files = list(results_dir.rglob("*.pickle"))
results_file = files[-1]
with open(results_file, "rb") as infile:
    results = pickle.loads(infile.read())

train_layer_emb = {}
test_layer_emb = {}
dataset_names = results.keys()
l2_label_mapping = {"regular_chat": -1, "system_prompt_clinical": 0, "privacy_inference": 1, "user_prompt": 2}

logger.info("Preprocessing results")
for layer in args.layer_number:
    train_layer_emb[layer] = defaultdict(dict)
    test_layer_emb[layer] = defaultdict(dict)
    for name in dataset_names:
        match = re.match(r"([a-zA-Z0-9_]+)_(train|test)", name)
        if match:
            category = match.group(1)
            split = match.group(2)
        else:
            raise ValueError(f"Invalid key format: {name}")
        dataset = results[name]
        if split == "train":
            train_layer_emb[layer][category]["ff_rep"] = np.concatenate([rep[layer] for rep in dataset['ff_rep']])
            train_layer_emb[layer][category]["label"] = np.concatenate([np.repeat(label, rep[layer].shape[0]) for label, rep in zip(dataset['label'], dataset['ff_rep'])])
            train_layer_emb[layer][category]["l2_label"] = np.where(train_layer_emb[layer][category]["label"] == 1, l2_label_mapping[category], -100)
        else:
            test_layer_emb[layer][category]["ff_rep"] = np.concatenate([rep[layer] for rep in dataset['ff_rep']])
            test_layer_emb[layer][category]["label"] = np.concatenate([np.repeat(label, rep[layer].shape[0]) for label, rep in zip(dataset['label'], dataset['ff_rep'])])
            test_layer_emb[layer][category]["l2_label"] = np.where(test_layer_emb[layer][category]["label"] == 1, l2_label_mapping[category], -100)

logger.info("Generating classifier results for each layer")
classifier_results = {}
for layer in args.layer_number:
    logger.info(f"Layer {layer}" + "="*20)
    train_features = np.concatenate([fl["ff_rep"] for fl in train_layer_emb[layer].values()])
    train_labels = np.concatenate([fl["label"] for fl in train_layer_emb[layer].values()])
    train_l2_labels = np.concatenate([fl["l2_label"] for fl in train_layer_emb[layer].values()])
    test_features = np.concatenate([fl["ff_rep"] for fl in test_layer_emb[layer].values()])
    test_labels = np.concatenate([fl["label"] for fl in test_layer_emb[layer].values()])
    test_l2_labels = np.concatenate([fl["l2_label"] for fl in test_layer_emb[layer].values()])
    if args.hierarchical:
        layer_acc, layer_auc = hierachical_gen_classifier_roc(train_features, train_labels, train_l2_labels, 
                                              test_features, test_labels, test_l2_labels)
    else:
        layer_auc, layer_acc = gen_classifier_roc(train_features, train_labels, test_features, test_labels)
    logger.info(f"Layer {layer}: ROC AUC: {layer_auc}, Accuracy: {layer_acc}")