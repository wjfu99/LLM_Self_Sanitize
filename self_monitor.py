import pickle
from pathlib import Path
import numpy as np
import scipy as sp

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import random

from tqdm import tqdm

gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
batch_size = 128
dropout_mlp = 0.5
dropout_gru = 0.25
learning_rate = 1e-4
weight_decay = 1e-2

inference_results = list(Path("./results/").rglob("*.pickle"))
print (inference_results)
    
    
class FFHallucinationClassifier(torch.nn.Module):
    def __init__(self, input_shape, dropout = dropout_mlp):
        super().__init__()
        self.dropout = dropout
        
        self.linear_relu_stack =torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 3)
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class RNNHallucinationClassifier(torch.nn.Module):
    def __init__(self, dropout=dropout_gru):
        super().__init__()
        hidden_dim = 128
        num_layers = 4
        self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, seq):
        gru_out, _ = self.lstm(seq)
        return self.linear(gru_out)[-1, -1, :]
    
    
def gen_classifier_roc(inputs, numeric_label):
    X_train, X_test, y_train, y_test = train_test_split(inputs, numeric_label, test_size = 0.2, random_state=123)
    classifier_model = FFHallucinationClassifier(X_train.shape[1]).to(device)
    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(torch.long).to(device)
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(torch.long).to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for _ in range(1001):
        optimizer.zero_grad()
        sample = torch.randperm(X_train.shape[0])[:batch_size]
        pred = classifier_model(X_train[sample])
        loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
        loss.backward()
        optimizer.step()
    classifier_model.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(classifier_model(X_test), dim=1)
        prediction_classes = torch.argmax(pred, dim=1).cpu()
        return roc_auc_score(y_test.cpu(), pred.cpu(), multi_class='ovr'), (prediction_classes.numpy() == y_test.cpu().numpy()).mean()
    
    
all_results = {}


for idx, results_file in enumerate(tqdm(inference_results)):
    if results_file not in all_results.keys():
        try:
            del results
        except:
            pass
        try:
            classifier_results = {}
            with open(results_file, "rb") as infile:
                results = pickle.loads(infile.read())
            label = np.array(results['label'])
            label_encoder = LabelEncoder()
            numeric_label = label_encoder.fit_transform(label)

            # fully connected
            for layer in range(results['first_fully_connected'][0].shape[0]):
                layer_roc, layer_acc = gen_classifier_roc(np.stack([i[layer] for i in results['first_fully_connected']]), numeric_label)
                classifier_results[f'first_fully_connected_roc_{layer}'] = layer_roc
                classifier_results[f'first_fully_connected_acc_{layer}'] = layer_acc

            # attention
            for layer in range(results['first_attention'][0].shape[0]):
                layer_roc, layer_acc = gen_classifier_roc(np.stack([i[layer] for i in results['first_attention']]), numeric_label)
                classifier_results[f'first_attention_roc_{layer}'] = layer_roc
                classifier_results[f'first_attention_acc_{layer}'] = layer_acc
            
            all_results[results_file] = classifier_results.copy()
        except Exception as err:
            print(err)
            
print(all_results.keys())

for k,v in all_results.items():
    print(k, v)