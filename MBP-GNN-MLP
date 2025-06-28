import sys
import urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.nn import GINEConv, GATv2Conv, global_add_pool

# Download SMILES chemical dataset from Tox21
url = "https://tripod.nih.gov/tox21/challenge/download?id={}".format("nr-arsmiles")
urllib.request.urlretrieve(url, './input.smi')

# Load SMILES data into DataFrame
df = pd.read_csv('./input.smi', sep=r'\s+', header=None)
df.columns = ['chemical struct', 'id', 'activity']
df.to_csv('input.csv')
df = pd.read_csv('input.csv', index_col=0)

# Convert a single SMILES string into a molecule
chemStruct = df['chemical struct'][9315]
mol = Chem.MolFromSmiles(chemStruct)

# Create edge list from bonds
edges = []
for bond in mol.GetBonds():
  i = bond.GetBeginAtomIdx()
  j = bond.GetEndAtomIdx()
  edges.extend([(i,j), (j,i)])
edge_index = list(zip(*edges))  # Convert to tuple of lists

# Define atom and bond feature functions
def atom_feature(atom):
  return [atom.GetAtomicNum(), atom.GetDegree(), atom.GetNumImplicitHs(), atom.GetIsAromatic()]

def bond_feature(bond):
  return [bond.GetBondType(), bond.GetStereo()]

# Convert molecule into PyG Data object with features
node_features = [atom_feature(a) for a in mol.GetAtoms()]
edge_features = [bond_feature(b) for b in mol.GetBonds()]
g = Data(edge_index=torch.LongTensor(edge_index), x=torch.FloatTensor(node_features), edge_attr=torch.FloatTensor(edge_features), chemStructure=chemStruct, mol=mol)

# Convert SMILES string into PyG graph for model training
def chemStruct_to_pyg(chemStruct, y):
  mol = Chem.MolFromSmiles(chemStruct)
  # Filters out invalid SMILES strings
  if mol is None:
    return None
  id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
  atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]
  bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
  atom_features = [atom_feature(a) for a in mol.GetAtoms()]
  bond_features = [bond_feature(b) for b in bonds]
  return Data(edge_index=torch.LongTensor(list(zip(*atom_pairs))), x=torch.FloatTensor(atom_features), edge_attr=torch.FloatTensor(bond_features), y=torch.LongTensor([y]), mol=mol, chemStructure=chemStruct)

# Create a custom Dataset for molecules
class MyDataset(Dataset):
  def __init__(self, chemStructure, response):
    mols = [chemStruct_to_pyg(chemStruct, y) for chemStruct, y in \
             tqdm(zip(chemStructure, response), total=len(chemStructure))]
    self.X = [m for m in mols if m]  # Filter out invalid molecules

  def __getitem__(self, idx):
    return self.X[idx]

  def __len__(self):
    return len(self.X)

# Initialize dataset and split into train/valid/predictions using an 80/10/10 ratio
base_dataset = MyDataset(df['chemical struct'], df['activity'])
N = len(base_dataset)
M = N // 10
indices = np.random.permutation(range(N))
idx = {'train': indices[:8*M], 'valid': indices[8*M:9*M], 'predictions': indices[9*M:]}
modes = ['train', 'valid', 'predictions']
dataset = {m: Subset(base_dataset, idx[m]) for m in modes}
loader = {m: DataLoader(dataset[m], batch_size=200, shuffle=(m == 'train')) for m in modes}

# Build and train AttentiveFP GNN
node_dim = base_dataset[0].num_node_features
edge_dim = base_dataset[0].num_edge_features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2 channels are used as a binary, there are 3 layers each with 200 nodes.

model = AttentiveFP(out_channels=2, in_channels=node_dim, edge_dim=edge_dim, hidden_channels=200, num_layers=3, num_timesteps=2, dropout=0.2)
model = model.to(device)
train_epochs = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(loader['train']), epochs=train_epochs)
criterion = nn.CrossEntropyLoss()
accuracy_curve = defaultdict(list)

# Define training loop
def train(loader):
  total_loss = total_examples = 0
  # Iterates over batches of graphs.
  for data in loader:
    data = data.to(device)
    optimizer.zero_grad()
    # Computes model predictions
    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    loss = criterion(out, data.y)
    # Calculates and backpropagates the loss.
    loss.backward()
    # Optimizer updates weights.
    optimizer.step()
    # Scheduler updates learning rate.
    scheduler.step()
    total_loss += loss.item()
    total_examples += data.num_graphs
  return total_loss / total_examples

# Define validation/testing loop
def test(loader):
  with torch.no_grad():
    total_loss = total_examples = 0
    for data in loader:
      data = data.to(device)
      out = model(data.x, data.edge_index, data.edge_attr, data.batch)
      loss = criterion(out, data.y)
      total_loss += loss.item()
      total_examples += data.num_graphs
  return total_loss / total_examples

# Get predictions from model
def predict(loader):
  # Disables gradient tracking for memory efficiency.
  with torch.no_grad():
    y_pred = []
    y_true = []
    for data in loader:
      data = data.to(device)
      out = model(data.x, data.edge_index, data.edge_attr, data.batch)
      _, predicted = torch.max(out.data, 1)
      y_true.extend(data.y.cpu().numpy())
      y_pred.extend(predicted.cpu().numpy())
  return y_true, y_pred

# Train AttentiveFP and track learning curve
best_val = float("inf")
learn_curve = defaultdict(list)
func = {'train': train, 'valid': test, 'predictions': test}
for epoch in tqdm(range(1, train_epochs+1)):
  loss = {mode: func[mode](loader[mode]) for mode in modes}
  for mode in modes:
    learn_curve[mode].append(loss[mode])
  if loss['valid'] < best_val:
    torch.save(model.state_dict(), 'best_val.model')
  if epoch % 20 == 0:
    print(f'Epoch: {epoch:03d} Loss: ' + ' '.join(['{} {:.6f}'.format(m, loss[m]) for m in modes]))

# Load best model and evaluate it
model.load_state_dict(torch.load('best_val.model'))
for m in ['valid', 'predictions']:
  # Use predict to get true and predicted labels.
  y_true, y_pred = predict(loader[m])
  for metric in [accuracy_score, balanced_accuracy_score, roc_auc_score]:
    print("{} {} {:.3f}".format(m, metric.__name__, metric(y_true, y_pred)))

# Utility function to create different GNN layers
def MyConv(node_dim, edge_dim, arch='GIN'):
  # It could return a GIN-style convolution with edge features.
  if arch == 'GIN':
    h = nn.Sequential(nn.Linear(node_dim, node_dim, bias=True))
    return GINEConv(h, edge_dim=edge_dim)
  # Or it could return a graph attention layer, integrating edge attributes.
  elif arch == 'GAT':
    return GATv2Conv(node_dim, node_dim, edge_dim=edge_dim)

# Custom GNN model definition
class MyGNN(nn.Module):
  def __init__(self, node_dim, edge_dim, arch, num_layers=3):
    super().__init__()
    self.convs = nn.ModuleList([MyConv(node_dim, edge_dim, arch) for _ in range(num_layers)])

  def forward(self, x, edge_index, edge_attr):
    for conv in self.convs:
      x = conv(x, edge_index, edge_attr)
      x = F.leaky_relu(x)
    return x

ptable = Chem.GetPeriodicTable()
# The loop finds the maximum atomic number supported
for i in range(200):
  try:
    s = ptable.GetElementSymbol(i)
  except:
    print(f'max id {i-1} for {s}')
    break
ptable.GetElementSymbol(i-1)

# Final neural network combining GNN + MLP head
class MyFinalNetwork(nn.Module):
  def __init__(self, node_dim, edge_dim, arch, num_layers=3, encoding='onehot'):
    super().__init__()
    self.encoding = encoding
    if encoding != 'onehot':
      self.atom_encoder = nn.Embedding(num_embeddings=119, embedding_dim=64)
      self.bond_encoder = nn.Embedding(num_embeddings=22, embedding_dim=8)
      node_dim = (node_dim-1) + 64
      edge_dim = (edge_dim-1) + 8
    else:
      node_dim = (node_dim-1) + 119
      edge_dim = (edge_dim-1) + 22

    self.gnn = MyGNN(node_dim, edge_dim, arch, num_layers=num_layers)

    embed_dim = int(node_dim / 2)
    self.head = nn.Sequential(nn.BatchNorm1d(node_dim), nn.Dropout(p=0.5), nn.Linear(node_dim, embed_dim), nn.ReLU(), nn.BatchNorm1d(embed_dim), nn.Dropout(p=0.5), nn.Linear(embed_dim, 2))
  def forward(self, x, edge_index, edge_attr, batch):
    # onehot uses fixed vectors to represent atom/bond types.
    if self.encoding == 'onehot':
      x0 = F.one_hot(x[:, 0].to(torch.int64), num_classes=119)
      edge_attr0 = F.one_hot(edge_attr[:, 0].to(torch.int64), num_classes=22)
    else:
      x0 = self.atom_encoder(x[:, 0].int())
      edge_attr0 = self.bond_encoder(edge_attr[:, 0].int())
    # Concatenate encoded types with other features.
    x = torch.cat([x0, x[:, 1:]], dim=1)
    edge_attr = torch.cat([edge_attr0, edge_attr[:, 1:]], dim=1)
    node_out = self.gnn(x, edge_index, edge_attr)
    graph_out = global_add_pool(node_out, batch)
    return self.head(graph_out)
  
# Train final custom GNN model
model = MyFinalNetwork(node_dim, edge_dim, arch='GAT', num_layers=3, encoding='embedding')
model = model.to(device)
train_epochs = 200
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(loader['train']), epochs=train_epochs)
criterion = nn.CrossEntropyLoss()
best_val = float("inf")
learn_curve = defaultdict(list)

# Training loop for custom GNN
for epoch in tqdm(range(1, train_epochs+1)):
  loss = {mode: func[mode](loader[mode]) for mode in modes}
  for mode in modes:
    learn_curve[mode].append(loss[mode])
  if 'valid' in loader:
    y_true, y_pred = predict(loader['valid'])
    current_valid_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    accuracy_curve['valid'].append(current_valid_balanced_accuracy)
  if 'predictions' in loader:
    y_true, y_pred = predict(loader['predictions'])
    current_predictions_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    accuracy_curve['predictions'].append(current_predictions_balanced_accuracy)
  if loss['valid'] < best_val:
    torch.save(model.state_dict(), 'best_val.model')
  if epoch % 20 == 0:
    print(f'Epoch: {epoch:03d} Loss: ' + ' '.join(['{} {:.6f}'.format(m, loss[m]) for m in modes]))

# Plot the training and validation loss
fig, ax = plt.subplots()
for m in modes:
  ax.plot(learn_curve[m], label=m)
ax.legend()
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.set_yscale('log')
plt.show()

# Plot the training and accuracy.
fig, ax = plt.subplots()
for m in ['valid', 'predictions']:
    ax.plot(accuracy_curve[m], label=m)
ax.set_xlabel('Epochs')
ax.set_ylabel('Balanced Accuracy')
ax.legend()
plt.show()

# Load the GNN model and report final performance
model.load_state_dict(torch.load('best_val.model'))
for m in ['valid', 'predictions']:
  y_true, y_pred = predict(loader[m])
  for metric in [accuracy_score, balanced_accuracy_score, roc_auc_score]:
    print("{} {} {:.3f}".format(m, metric.__name__, metric(y_true, y_pred)))
