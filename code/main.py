import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import array
import sys, copy, math, time, pdb
import os
import os.path as osp

import random
import argparse
import networkx as nx
#from scipy import interp
from util_functions import *
from torch_geometric import loader

#from torch_geometric.data import DataLoader
from model import Net
from itertools import chain
from torch.optim import *
from sklearn import metrics
from sklearn.metrics import average_precision_score,p$
from openpyxl import load_workbook
import gc
import pandas as pd
from sklearn.model_selection import StratifiedKFold,t$
# from pytorchtools import EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser(description='Link Prediction')
    # general settings
    parser.add_argument('--dataset', default='NPInter4158', help='network name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--initialLearningRate', default=0.0001,type=float, help='Initial learning rate')
    parser.add_argument('--l2WeightDecay', default=0.001, type=float, help='L2 weight')
    parser.add_argument('--epochNumber', default=60,type=int, help='number of training epoch')
    parser.add_argument('--batchSize', default=128, type=int, help='batch size')

    return parser.parse_args()

args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

# default program settings
TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"
# get the path
script_dir, script_name = osp.split(os.path.abspath(sys.argv[0]))
parent_dir = osp.dirname(script_dir)

# set paths of data, results and program parameters
DATA_BASE_PATH = parent_dir + '/data/'
RESULT_BASE_PATH = parent_dir + '/result/'
# set result save path
result_save_path = RESULT_BASE_PATH + args.dataset + "/" + args.dataset + time.strftime(TIME_FORMAT, time.localtime()) + "/"
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)


K_FOLD = 
PATIENCES = 10
device = torch.device('')

def read_interaction_dataset(dataset_path, dataset_name):
    return interaction_list, negative_interaction_list, ncRNA_list, protein_list, ncRNA_name_index_dict, protein_name_index_dict,\
    ncRNA_name_serialnumber_dict,protein_name_serialnumber_dict, set_interactionKey, set_negativeInteractionKey,sample_name_serialnumber_dict,all_interaction,\
    pos_interaction,neg_interaction

def negative_interaction_generation():
  
def output_information(path:str, information:dict):
  
def get_key(dict,value):
    return [k for k, v in dict.items() if v == value]

def serialnumber_transfer_name(all_interaction):
    return all_interaction_name

def networkx_format_network_generation(interaction_list, ncRNA_list, protein_list):
    return G

def output_edgelist_file(G, output_path):
 
def get_k_fold_data(k, data):
    return train_data, test_data

def train(model,device,train_loader,optimizer):
    return avg_loss, all_targets, all_pred, auc

def test(model,device,test_loader):
   return avg_loss, all_targets, all_pred, all_scores,auc

def performance(tp,tn,fp,fn):
    return ACC,Sen, Spe,Pre,MCC,F1


if __name__ == '__main__':
    start = time.time()
    print("start:{}".format(start))
    result_file = open(result_save_path + 'result.txt', 'w')
   
    interaction_dataset_path = 
    interaction_list, negative_interaction_list, ncRNA_list, protein_list, ncRNA_name_index_dict, protein_name_index_dict, ncRNA_name_serialnumber_dict, \
    protein_name_serialnumber_dict, set_interactionKey, set_negativeInteractionKey, sample_name_serialnumber_dict, all_interaction, \
    pos_interaction, neg_interaction = read_interaction_dataset()
   
    negative_interaction_generation()
    all_interaction_name = 
    all_interaction_name = 
    all_interaction_name.to_csv()
    all_interaction = pd.DataFrame()

    train_data, test_data = get_k_fold_data()
    G = networkx_format_network_generation()
    adj = nx.adjacency_matrix()
    node_feature = pd.read_csv().values
 
    print('\n\nK-fold cross validation processes:\n')
    result_file.write(f'{K_FOLD}-fold cross validation processes:\n')

    i = 0
    for fold in range(K_FOLD):
     result_file.write("ACC:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, F1:{}, AUC:{}".
                          format(test_acc, test_pre, test_sen, test_spe, test_MCC, test_F1, test_auc) + '\n')


    model.train()

    end = time.time()
    print("total {} seconds".format(end - start))

