import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import os.path as osp
import pickle as cp
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
import warnings
import multiprocessing as mp
import torch

from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx

# get the path
script_dir, script_name = osp.split(os.path.abspath(sys.argv[0]))
parent_dir = osp.dirname(script_dir)

# set paths of data, results and program parameters
DATA_BASE_PATH = 
RESULT_BASE_PATH = 

class GNNGraph(object):

def links2subgraphs(A, temp_train_data, temp_test_data ,temp_train_label,temp_test_label,node_feature): #h=2, max_nodes_per_hop=None,
   return train_graphs, test_graphs, max_n_label['value']

def links2subgraphs1(A, temp_train_data, temp_test_data ,temp_train_label,temp_test_label,node_feature):
   return train_graph, test_graph,max_n_label['value']
  
def links2subgraphs2(A, temp_train_data, temp_test_data ,temp_train_label,temp_test_label,node_feature):
    return train_graph, test_graph,max_n_label['value']

class Graph(object):
   
def nx_to_PyGGraph(batch_graphs,max_n_label):
   return DATA

def subgraph_extraction_labeling1(ind, A,g_label,node_feature,h=2, max_nodes_per_hop=30):
    return g, g_label,labels.tolist() , node_feature

def subgraph_extraction_labeling(ind, A,g_label,node_feature,h=2, max_nodes_per_hop=30):
    return g, g_label,labels.tolist() , node_feature

def neighbors(fringe, A):
    return res

def node_label(subgraph):
   return labels


def to_linkgraphs(batch_graphs, max_n_label):
   return graphs

def edge_fea(graph, max_n_label):
   return x

def to_undirect(edges, edge_fea):
    return np.concatenate([rp, pr], axis=1), fea_body

class Node:
  
class ncRNA:
 
class Protein:
  
class ncRNA_Protein_Interaction:
   
def printN(pred, target):
     return TP,TN,FP,FN

def true_positive(pred, target):
    return ((pred == 1) & (target == 1)).sum().clone().detach().requires_grad_(False)


def true_negative(pred, target):
    return ((pred == 0) & (target == 0)).sum().clone().detach().requires_grad_(False)

def false_positive(pred, target):
    return ((pred == 1) & (target == 0)).sum().clone().detach().requires_grad_(False)


def false_negative(pred, target):
    return ((pred == 0) & (target == 1)).sum().clone().detach().requires_grad_(False)

def precision(pred, target):
   return out


def sensitivity(pred, target):
    return out

def specificity(pred, target):
   return out

def MCC(pred,target):
   return out

def accuracy(pred, target):
   return out

def FPR(pred, target):
  return out

def TPR(pred, target):
   return out

def AUC(label, prob):
    return roc_auc_score(label, prob)(base) [SurbhiMistry@param-hpc-iar utils]$
