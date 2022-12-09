import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from util_functions import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool


parser = argparse.ArgumentParser(description='GNN4REL')
# general settings
parser.add_argument('--file-name', default=None, help='dataset file name')
parser.add_argument('--links-name', default=None, help='links name')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; \
                    by default use all cpu cores to extract subgraphs in parallel')
# model settings
parser.add_argument('--hop', default=1, metavar='S',
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')

parser.add_argument('--use-test', action='store_true', default=False,
                    help='whether to read testing separtly')
parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

parser.add_argument('--num-links', type=int, default=1000,
                        help='max_training_path (default: 1000)')
parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
parser.add_argument('--filename', type = str, default = "",
                                        help='output file')

args = parser.parse_args()
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
print(args)
args.hop = int(args.hop)
'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
val_pos, val_neg, train_pos, test_pos,train_neg,test_neg,link_pos = None,None, None, None,None,None,None
if args.links_name is not None:
    print("The link file was provided")
    args.links_dir = os.path.join(args.file_dir, './data/{}/{}'.format(args.file_name,args.links_name))
    links_idx = np.loadtxt(args.links_dir, dtype=int)
    links_pos = (links_idx[:, 0], links_idx[:, 1])
file_lines_test=[]
if args.use_test:
    with open('./data/{}/paths_test.txt'.format(args.file_name), 'r') as fhand:
        file_lines_test = [line[:-1] for line in fhand if line.strip() != ''] # remove the last character '\n'. **Remove empty lines**.
with open('./data/{}/paths.txt'.format(args.file_name), 'r') as fhand:
    file_lines = [line[:-1] for line in fhand if line.strip() != ''] # remove the last character '\n'. **Remove empty lines**.
feat=[]
count=[]
labels=[]
feats_test = np.loadtxt('./data/{}/feat.txt'.format(args.file_name), dtype='int32')
#Later, we remove the driving and degree features. They will be dropped
labels_ = np.loadtxt('./data/{}/label.txt'.format(args.file_name), dtype='float32')
labels_test=[]
if args.use_test:
    labels_test = np.loadtxt('./data/{}/label_test.txt'.format(args.file_name), dtype='float32')
count = np.loadtxt('./data/{}/count.txt'.format(args.file_name))
arr1inds = count.argsort()
attributes = feats_test[arr1inds[0::]]
max_idx = np.max(links_idx)
net = ssp.csc_matrix((np.ones(len(links_idx)), (links_idx[:, 0], links_idx[:, 1])), shape=(max_idx+1, max_idx+1) )
net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
B=net.copy() # a matrix with direction
B.eliminate_zeros()
net[links_idx[:, 1], links_idx[:, 0]] = 1  # add symmetric edges
net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
print("Done creating the net. Now time for sampling")
A = net.copy()  # the observed network
A.eliminate_zeros()
node_information = None
node_information = attributes
print(str(len(labels_)))

dict_labels = {}
A_array=np.array(file_lines)
zip_iterator = zip(A_array,labels_)
dict_labels = dict(zip_iterator)

dict_labels_test = {}

if args.use_test:
    A_array=np.array(file_lines_test)
    zip_iterator_test = zip(A_array,labels_test)
    dict_labels_test = dict(zip_iterator_test)
'''Train and apply the model'''

np.random.shuffle(file_lines)
file_lines=file_lines[:args.num_links]
train_graphs=None
test_graphs=None
if args.use_test:
    print("Testing was given!")
    print(file_lines_test)
    print(dict_labels_test)
    train_graphs, test_graphs = gates2subgraphs(
        A,
        B,
        file_lines,
        dict_labels,
        file_lines_test,
        dict_labels_test,
        args.hop,
        node_information,
        args.no_parallel,
    )
else:
    print("Testing was not given!")
    train_graphs,test_graphs = gates2subgraphs(
        A,
        B,
        file_lines,
        dict_labels,
        None,
        None,
        args.hop,
        node_information,
        args.no_parallel,
    )
print("Done Testing Enclosing Subgraph Extraction")
random.shuffle(train_graphs)
val_num = int(0.1 * len(train_graphs))
val_dataset = train_graphs[:val_num]
train_graphs = train_graphs[val_num:]
random.shuffle(train_graphs)
test_dataset=None
train_dataset=None
if test_graphs is None:
    print("Normal shuffling procedure")
    test_num = int(0.1 * len(train_graphs))
    test_dataset = train_graphs[:test_num]
    train_dataset = train_graphs[test_num:]
else:
    train_dataset=train_graphs
    test_dataset=test_graphs
print("Training number is "+str(len(train_dataset))+", Validation number is "+str(len(val_dataset))+", Testing number is"+str(len(test_dataset)))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #batch size was 32. does it matter now?
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

maxx=0


for data in val_dataset:
    print(data.edge_index[1])
    print(data.num_nodes)
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    if max(d) > maxx:
        maxx=max(d).item()
for data in test_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    if max(d) > maxx:
        maxx=max(d).item()
for data in train_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    if max(d) > maxx:
        maxx=max(d).item()
print("Max degree is "+str(maxx))
maxx=maxx+1
deg = torch.zeros(int(maxx), dtype=torch.long)
full_dataset=train_dataset+test_dataset+val_dataset
for data in full_dataset: #train_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #I have replaced 75 by 32.
        self.node_emb = Embedding(23, 32)#here
        self.edge_emb = Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(deg=deg, in_channels=32, out_channels=32, edge_dim=50,
                           aggregators=aggregators, scalers=scalers, towers=4, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(32))#here
        #here
        self.mlp = Sequential(Linear(32, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        print("This is the length of x")
        print(len(x))

        print("This is the length of x squeeze")
        print(len(x.squeeze()))
        print("This is the max X")
        print(max(x))
        x = self.node_emb(x.squeeze())

        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)


def train_log(epoch):
    model.train()
    total_loss = 0
    for data in train_loader:

        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        losst = torch.nn.MSELoss()
        print(out.squeeze())
        print(data.y)
        loss=torch.sqrt(losst(torch.log(out.squeeze() + 1), torch.log(data.y + 1)))
        print(loss)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def train(epoch):
    model.train()
    total_loss = 0
    print("In Training function")
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        print(data.edge_index)
        print(data.x)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)
@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)

def print_test(loader):
    model.eval()
    total_error = 0
    total_per_error=0
    total_per=0
    RMSE=0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        print("Data Label is ")
        print(data.y)
        print("Data Prediction is ")
        print(out.squeeze())
        total_error += (out.squeeze() - data.y).abs().sum().item()
        diff=(out.squeeze() - data.y).abs()
        total_per +=(diff / data.y).abs().sum().item()
        RMSE +=diff.pow(2).sum().item()
    total_per_error=total_per*100/len(loader.dataset)
    return total_error / len(loader.dataset) , total_per_error, math.sqrt(RMSE/len(loader.dataset))

best_loss = None
best_epoch = None
for epoch in range(1, 301):
    loss = train(epoch)
    val_mae = test(val_loader)
    test_mae = test(test_loader)
    scheduler.step(val_mae)
    if not args.filename == "":
        with open(args.filename, 'a') as f:
            f.write("Epoch: %d Training loss: %f Val MAE: %f Test MAE: %f" % (epoch, loss, val_mae, test_mae))
            f.write("\n")
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')

    if best_loss is None:
        best_loss = val_mae
    if val_mae <= best_loss:
        print("Epoch "+str(epoch)+"Is better, performing testing")
        best_loss = val_mae
        best_epoch = epoch
        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write("Epoch: %d is best Validation Loss: %f Val MAE: %f Test MAE: %f" % (epoch, loss, val_mae, test_mae))
                f.write("\n")

        ignore, ignore2,RMSD = print_test(test_loader)

        print("Total RMSD error is %f" % (RMSD))
        print("Total percentage error is %f" % (ignore2))
