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
from utils import FFSelfMonitor
from tqdm import tqdm
import argparse
import logging
import re

parser = argparse.ArgumentParser(description="Self-monitoring model")
parser.add_argument("--embeddings_dir", type=str, default="./privacy_datasets/embeddings", help="The directory to save the embeddings")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="The model name to use")
parser.add_argument("--class_num", type=int, default=4, help="Number of classes")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--dropout_mlp", type=float, default=0.5, help="Dropout rate for MLP")
parser.add_argument("--dropout_gru", type=float, default=0.25, help="Dropout rate for GRU")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")
parser.add_argument("--layer_number", type=int, nargs="+", default=[32, 33, 34, 35, 36], help="The layer number to use")
parser.add_argument("--hierarchical", action="store_true", help="Whether to use hierarchical self-monitoring")

args = parser.parse_args()

gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

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
        loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
        loss.backward()
        optimizer.step()
    classifier_model.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(classifier_model(X_test), dim=1)
        prediction_classes = torch.argmax(pred, dim=1).cpu()
    torch.save(classifier_model.state_dict(), "./self_monitor_models/classifier_model_layer{}.pth".format(layer))
    prediction_classes = (pred[:,1]>0.5).type(torch.long).cpu()
    return roc_auc_score(y_test.cpu(), pred[:,1].cpu()), (prediction_classes.numpy()==y_test.cpu().numpy()).mean()
        

results_dir = Path(f"{args.embeddings_dir}/{args.model_name}")
files = list(results_dir.rglob("*.pickle"))
results_file = files[-1]
with open(results_file, "rb") as infile:
    results = pickle.loads(infile.read())

train_layer_emb = {}
test_layer_emb = {}
dataset_names = results.keys()

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
        else:
            test_layer_emb[layer][category]["ff_rep"] = np.concatenate([rep[layer] for rep in dataset['ff_rep']])
            test_layer_emb[layer][category]["label"] = np.concatenate([np.repeat(label, rep[layer].shape[0]) for label, rep in zip(dataset['label'], dataset['ff_rep'])])

classifier_results = {}
# fully connected
for layer in args.layer_number:
    train_features = np.concatenate([fl["ff_rep"] for fl in train_layer_emb[layer].values()])
    train_labels = np.concatenate([fl["label"] for fl in train_layer_emb[layer].values()])
    test_features = np.concatenate([fl["ff_rep"] for fl in test_layer_emb[layer].values()])
    test_labels = np.concatenate([fl["label"] for fl in test_layer_emb[layer].values()])
    layer_roc, layer_acc = gen_classifier_roc(train_features, train_labels, test_features, test_labels)
    classifier_results[f'first_fully_connected_roc_{layer}'] = layer_roc
    classifier_results[f'first_fully_connected_acc_{layer}'] = layer_acc

print(classifier_results)