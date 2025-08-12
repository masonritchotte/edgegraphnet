import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

import torch
torch.manual_seed(420)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

"""
Initializing parameters
"""
undirected = True
num_epochs = 200
batch_size = 16
train_per = 0.8
test_per = 0.1
"""
Data Intaking
"""
adj_file_path = '/home/mzr19001/edge graph net/data/embedding.npy'
expression_data_path = '/home/mzr19001/edge graph net/data/tcga/top500_expression_data.csv'
labels_data_path = '/home/mzr19001/edge graph net/data/tcga/tcga_labels.csv'

adj_file = np.load(adj_file_path, allow_pickle=True)
expression_data = pd.read_csv(expression_data_path, header=0, index_col=0)
labels_data = pd.read_csv(labels_data_path, header=0, index_col=0)

"""
Data Processing
"""
edge_attr = adj_file[:, 2]
source_nodes = adj_file[:, 0]
target_nodes = adj_file[:, 1]

if undirected:
    # Duplicate for undirected graph
    edge_attr = np.concatenate((edge_attr, edge_attr), axis=0)
    temp = source_nodes.copy()
    source_nodes = np.concatenate((source_nodes, target_nodes), axis=0)
    target_nodes = np.concatenate((target_nodes, temp), axis=0)
    edges = np.stack((source_nodes, target_nodes), axis=1)
else:
    edges = np.stack((source_nodes, target_nodes), axis=1)

# Lay matrix on side for COO format
edges_T = edges.T.copy().astype(np.int64)

# print("Edge attribute shape:", edge_attr.shape)
# print("First edge attribute:", edge_attr[0])
# print("Source nodes shape:", source_nodes.shape)
# print("Target nodes shape:", target_nodes.shape)

# print(source_nodes[0])
# print(target_nodes[0])
# print(source_nodes[len(temp)])
# print(target_nodes[len(temp)])
# print(edge_attr[0] == edge_attr[len(temp)])  # Should be True

edge_index = torch.tensor(edges_T, dtype=torch.long)
edge_attr = np.stack(edge_attr).astype(np.float64)
# print("{:.17f} {:.17f} {:.17f}".format(edge_attr[0][0].item(), edge_attr[0][1].item(), edge_attr[0][2].item()))

# Prepare features and labels
features = torch.tensor(expression_data.values, dtype=torch.float)
le = LabelEncoder()
labels = torch.tensor(le.fit_transform(labels_data.values.squeeze()), dtype=torch.long)

print("Feature mean", "Feature std")
print(features.mean(), features.std())

# Split indices for train, val, test
num_samples = features.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)
train_end = int(train_per * num_samples)
test_end = train_end + int(test_per * num_samples)
train_idx = indices[:train_end]
test_idx = indices[train_end:test_end]
val_idx = indices[test_end:]
x_train, y_train = features[train_idx], labels[train_idx]
x_val, y_val = features[val_idx], labels[val_idx]
x_test, y_test = features[test_idx], labels[test_idx]

# Convert to tensors
x_train = torch.tensor(features[train_idx], dtype=torch.float32)
y_train = torch.tensor(labels[train_idx], dtype=torch.long)
x_val = torch.tensor(features[val_idx], dtype=torch.float32)
y_val = torch.tensor(labels[val_idx], dtype=torch.long)
x_test = torch.tensor(features[test_idx], dtype=torch.float32)
y_test = torch.tensor(labels[test_idx], dtype=torch.long)

# Create TensorDataset for each split
train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)
test_data = TensorDataset(x_test, y_test)

# Create DataLoader for each split
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)