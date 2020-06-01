#!/usr/bin/env python  
#-*- coding:utf-8 _*-  


import os
import sys
import re
import time
import json
import math
import random
import pickle
import logging
import argparse
import subprocess

from collections import defaultdict

import scipy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T


from tqdm import tqdm

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid

from torch.utils.tensorboard import SummaryWriter


CUR_DIR = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=13, type=int, help='seed(default 13)')
parser.add_argument('--dirpath', default=CUR_DIR, help='dirpath')
parser.add_argument('--dataset', default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], help='dataset')
parser.add_argument('--devices', default='cuda:0',  help='deviece: e.g.(cuda:0)')
parser.add_argument('--gnn', default='gcn', choices=['gcn', 'gat'], required=True, help='GCN or GAT')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--lambda1', type=float, default=1.0, help='loss lambda')
parser.add_argument('--lambda2', type=float, default=0.0, help='z1, z2, lambda')
parser.add_argument('--label_num', type=int, default=20, help='more label')

parser.add_argument('--early_stopping1', default=20, type=int, help='pretrain_early_stop')
parser.add_argument('--early_stopping2', default=1000, type=int, help='early_stop')
parser.add_argument('--runs', type=int, default=30)


args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)


MODEL_NAME = args.gnn
DATASET = args.dataset
LAMBDA1 = args.lambda1
LAMBDA3 = 1

assert LAMBDA1 >= 0.0 and LAMBDA1 <= 1.0, 'LAMBDA1 must in [0, 1]'

LAMBDA2 = args.lambda2
LABEL_NUM = args.label_num
RANDOM_SPLITS = args.random_splits

DEVICE = torch.device(args.devices)
DEBUG = args.debug
RUNS = 1 if DEBUG else args.runs

if MODEL_NAME == 'gat' and args.dataset == 'PubMed':
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.001
    OUTPUT_HEADS = 8
else:
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.0005
    OUTPUT_HEADS = 1


if MODEL_NAME == 'gcn':
    EARLY_STOPPING1 = 10
    EPOCHES1 = 200
    EPOCHES2 = 1000
else:
    EARLY_STOPPING1 = 100
    EPOCHES1 = 1000
    EPOCHES2 = 2000


writer = SummaryWriter(comment=f'{DATASET}-{MODEL_NAME}-{LAMBDA1}-{LAMBDA2}-{LABEL_NUM}-{RANDOM_SPLITS}')


class GCNNet(nn.Module):
    def __init__(self, dataset):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, training=None):

        x = F.relu(self.conv1(x, edge_index))
        training = self.training if training == None else training
        x = F.dropout(x, p=0.5, training=training)
        x = self.conv2(x, edge_index)
        return x



class GATNet(nn.Module):
    def __init__(self, dataset):
        super(GATNet, self).__init__()

        self.conv1 = GATConv(
            dataset.num_features,
            8,
            heads=8,
            dropout=0.6)

        self.conv2 = GATConv(
            8 * 8,
            dataset.num_classes,
            heads=OUTPUT_HEADS,
            concat=False,
            dropout=0.6)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, training=None):
        training = self.training if training == None else training
        x = F.dropout(x, p=0.6, training=training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=training)
        x = self.conv2(x, edge_index)
        return x



def output(outs, run):
    if DEBUG:
        writer.add_scalar("val_loss", outs['val_loss'], outs['epoch'])
        writer.add_scalar("val_acc",  outs['val_acc'] , outs['epoch'])
        writer.add_scalar("test_acc", outs['test_acc'], outs['epoch'])


class Net(nn.Module):
    def __init__(self, gnn, dataset):
        super(Net, self).__init__()
        self.gnn1 = gnn(dataset)

    def forward1(self, data):
        x = self.gnn1(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)

    def forward(self, data):
        x0, edge_index = data.x, data.edge_index
        x1 = self.gnn1(x0, edge_index, training=self.training)
        
        # 加上一个用邻接矩阵做标签传播的
        h0 = torch.softmax(x1, dim=1)
        node_num = len(data.y)
        adj_0 = torch.zeros([node_num,node_num]).to(DEVICE)
        adj_0[edge_index[0, :], edge_index[1, :]] = torch.ones(1).to(DEVICE)
        adj_0 = adj_0 + torch.diag(torch.ones([node_num]).to(DEVICE))
        # deg = torch.diag(adj_0.sum(dim=1).to(DEVICE))
        # deg_inv_sqrt = deg.pow(-0.5).to(DEVICE)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt = torch.diag(adj_0.sum(dim=1).pow(-0.5).to(DEVICE))
        adj_t = torch.mm(torch.mm(deg_inv_sqrt,adj_0),deg_inv_sqrt)
        x_finally = torch.mm(adj_t, h0)
        adj_0 = torch.mm(h0, h0.t())

        # adj_1_1 = torch.zeros(adj_1_0.shape).to(DEVICE)
        # adj_1_1[edge_index[0, :], edge_index[1, :]] = torch.ones(1).to(DEVICE)
        # adj_1_1 = adj_1_1 + torch.diag(torch.ones(adj_1_0.shape[0]).to(DEVICE))


        # topk = edge_index.size()[1] * LAMBDA3 
        # value, indices = torch.topk(adj_0.reshape(1, -1), LAMBDA3)
        # topk_v = value[0][-1]
        # adj_1_0 = torch.where(adj_0 > topk_v, torch.ones(1).to(DEVICE), torch.zeros(1).to(DEVICE))
        # adj_1_0 = adj_1_0 + torch.diag(torch.ones(adj_1_0.shape[0]).to(DEVICE))
        # adj_1 = torch.where(adj_1_0 > 0, torch.ones(1).to(DEVICE), torch.zeros(1).to(DEVICE))
        # adj_1_1 = torch.zeros(adj_1_0.shape).to(DEVICE)
        # adj_1_1[edge_index[0, :], edge_index[1, :]] = torch.ones(1).to(DEVICE)
        # adj_1 = adj_1_0 + adj_1_1
        # adj_1 = adj_0
        # h1 = torch.mm(adj_1, h0)

        ## equal to
        # tmp = torch.mm(h.t(), h)
        # h1 = torch.mm(h, tmp)

        # h0 = torch.softmax(x1, dim=1)
        # adj_0 = torch.mm(h0, h0.t())
        # h1 = torch.mm(adj_0, h0)
        # eps = 1e-12
        # h1 = h1 / (h1.sum(dim=1).view(-1, 1) + eps)
        # x_finally = LAMBDA1 * h1 + (1 - LAMBDA1) * h0
        # x_finally = x_finally / (x_finally.sum(dim=1).view(-1, 1) + eps)

        return adj_0, torch.log(x_finally)


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    def index_to_mask(index, size):
        mask = torch.zeros(size, dtype=torch.uint8, device=index.device)
        mask[index] = 1
        return mask

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([index[:20] for index in indices], dim=0)

    rest_index = torch.cat([index[20:] for index in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data



def read_dataset(dataset_name):
    dataset = Planetoid(root=f'./datas/{dataset_name}', name=dataset_name)
    dataset.transform = T.NormalizeFeatures()
    data = dataset[0]
    if RANDOM_SPLITS:
        data = random_planetoid_splits(data, dataset.num_classes)

    if LABEL_NUM < 20:
        index = defaultdict(list)
        for ind, i in enumerate(data.y[: dataset.num_classes * 20].numpy()):
            if len(index[i]) < LABEL_NUM:
                index[i].append(ind)

        ind = []
        for i in index:
            for j in index[i]:
                ind.append(j)
        data.train_mask = torch.zeros(data.train_mask.shape, dtype=torch.uint8)
        data.train_mask[ind] = 1

    return data, dataset



def evaluate(logits, data):
    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).int().sum().item() / mask.int().sum().item()
        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs



def main():

    res = []
    res0 = []
    if MODEL_NAME == 'gcn':
        gnn = GCNNet
    else:
        gnn = GATNet

    for run in range(RUNS):


        data, dataset = read_dataset(DATASET)
        model = Net(gnn, dataset).to(DEVICE)
        data = data.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        test_acc = 0
        val_best = {}
        best_val_acc = 0
        best_val_loss = float('inf')
        val_loss_history = []
        #
        for epoch in tqdm(range(1, EPOCHES1+1)):
            model.train()
            optimizer.zero_grad()
            rs = model.forward1(data)
            loss = F.nll_loss(rs[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model.forward1(data)

            outs = evaluate(logits, data)
            eval_info = outs
            eval_info['epoch'] = epoch
            output(eval_info, run)

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']

            val_loss_history.append(eval_info['val_loss'])
            if EARLY_STOPPING1 > 0 and epoch > EPOCHES1 // 2:
                tmp = torch.tensor(val_loss_history[-(EARLY_STOPPING1 + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        print('Pre train model test Acc', test_acc)
        res0.append(test_acc)

        test_acc = 0
        val_best = {}
        best_val_acc = 0
        best_val_loss = float('inf')
        val_loss_history = []

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for epoch in tqdm(range(1 + EPOCHES1, 1 + EPOCHES1 + EPOCHES2)):

            model.train()
            optimizer.zero_grad()
            adj0, rs = model.forward(data)
            loss = F.nll_loss(rs[data.train_mask], data.y[data.train_mask])

            ### 补一个监督loss
            mask = data.train_mask
            adj_pred = adj0[mask, :][:, mask]

            a = data.y[data.train_mask].cpu()
            b = torch.cat([torch.arange(0, a.shape[0]).long().unsqueeze(1), a.unsqueeze(1)], dim=1)
            b = b.long()

            c = torch.zeros(a.shape[0], a.max() + 1)
            c[b[:, 0], b[:, 1]] = torch.Tensor([1])
            adj_gth = torch.mm(c, c.t()).to(DEVICE)
            # loss1 = F.binary_cross_entropy(adj_pred, adj_gth)
            # loss2 = loss + LAMBDA2 * loss1
            loss2 = loss
            #print(loss, loss1, loss2)

            loss2.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                adj0, logits = model.forward(data)

            outs = evaluate(logits, data)
            eval_info = outs
            eval_info['epoch'] = epoch
            output(eval_info, run)

            if eval_info['val_acc'] >= best_val_acc:
                if eval_info['val_acc'] == best_val_acc and eval_info['val_loss'] > best_val_loss:
                    continue
                best_val_acc = eval_info['val_acc']
                best_val_loss = eval_info['val_loss']
                val_best = eval_info


        res.append(val_best['test_acc'])
        print('Our Model Acc: ', val_best['test_acc'], test_acc)
        #print(val_best)

    print(np.mean(res0), np.std(res0))
    print(np.mean(res), np.std(res))



if __name__ == '__main__':
    main()
