from __future__ import print_function
import torch
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import convert, to_undirected
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import multiprocessing as mp
from itertools import islice

def gates2subgraphs(A,B, train_nodes,  dict_labels, test_nodes=None, dict_labels_test=None, h=2, node_information=None, no_parallel=False):
    def helper(A,B, links, dict_labels_pass):
        print("Inside Helper!")
        g_list = []
        if no_parallel:
            for i in tqdm(links):
                print("Extracting subgraph i "+str(i))
                x, edge_index, edge_attr, ind = subgraph_extraction_labeling(dict_labels_pass[i], i, A, B,h, node_information)
                g_list.append(Data(x=x, y=torch.Tensor(np.array(dict_labels_pass[i])), edge_index=edge_index,edge_attr=edge_attr ))
            return g_list
        else:
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.map_async(
                parallel_worker,
                [(dict_labels_pass[i],i, A,B, h,  node_information) for i in links]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            g_list = [Data(x=x, y=torch.Tensor(np.array(dict_labels_pass[ind])), edge_index=edge_index,edge_attr=edge_attr ) for x, edge_index, edge_attr,ind in results]
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs = None
    test_graphs = None
    train_graphs = helper(A,B, train_nodes, dict_labels)
    if test_nodes is not None:
        print("Calling helper again with test nodes")
        print(test_nodes)
        test_graphs = helper(A,B, test_nodes, dict_labels_test)
    return train_graphs, test_graphs

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

def subgraph_extraction_labeling(label_f, path, A, B, h=2,node_information=None):
    print("Extracting subgraph for path "+path)
    dist = 0
    a_list = path.split()
    map_object = map(int, a_list)
    list_of_integers = list(map_object)
    nodes=set(list_of_integers)
    nodes = set(list_of_integers)
    visited = set(list_of_integers)
    fringe = set(list_of_integers)
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
    nodes=list(nodes)
    sub_size=len(nodes)
    subgraph = B[nodes, :][:, nodes] #for now, only keeping the directed adjacency

    features=None
    if node_information is not None:
        i=0
        for node in nodes:
            vector=[]
            vector=list(node_information[node])
            vector=vector[:-3] #drop the last three features
            indices_ = [i_ for i_, j_ in enumerate(vector) if j_ == 1]
            if i>0:
                features= torch.cat([features, torch.Tensor(indices_)], dim=0)
            else:
                features=  torch.Tensor(indices_)
            i=i+1

    features = features.long()
    features=torch.reshape(features, (len(features), 1))
    edge_index, edge_weight = convert.from_scipy_sparse_matrix(subgraph)
    print("This is edge weight")
    print(edge_weight)
    use_dir_new=False
    if use_dir_new:
        row, col = edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        d=torch.zeros(len(edge_weight))
        d=torch.add(d, 3)
        edge_weight = torch.cat([edge_weight, d], dim=0)
    edge_weight=torch.reshape(edge_weight, (len(edge_weight),1))

    print("Done extracting subgraph for path "+path)
    return features, edge_index.long(), edge_weight.long(), path

def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res
