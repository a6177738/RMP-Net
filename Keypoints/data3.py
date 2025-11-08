import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import os
import json


# Function to flip coordinates horizontally
def flip_coordinates(coords, max_x=512):
    return [(max_x - x, y) for x, y in coords]

def out_data():
    x = list()
    labels = list()
    arra = list()
    for i in range(99, 277):
        arra.append(i)
    train_index = arra

    for i in train_index:
        root = '../original_data_whole/'
        point_i = np.loadtxt(root + str(i) + '/norm_3cut.txt',dtype=float)[:28,:2]
        with open(root+str(i)+'/label.txt', "r") as f:
            label = int(f.read().strip())
        x.append(point_i)
        labels.append(label)
    x = torch.from_numpy(np.array(x,dtype = np.float32))  # 节点特征维度为 (28, 2)
    edge_index = np.loadtxt('../edge/edge3_index.txt')
    edge_index = torch.tensor(edge_index, dtype=torch.int)  # 边索引维度为 (2, 13)
    y = torch.tensor(labels, dtype=torch.float32)  # 图分类任务的类别标签

    return x,edge_index, y

def out_testdata():
    x = list()
    labels = list()
    test_index = [68, 199, 231, 41, 104, 203, 268, 272, 274, 148, 28, 61, 191, 129, 132, 139, 144, 147, 23, 153, 26, 155, 33, 162, 164, 166, 40, 168, 42, 43, 173, 176, 182, 62, 192, 71, 201, 207, 210, 215, 216, 225, 102, 233, 107, 109, 238, 239, 241, 116, 254, 249, 123, 252, 125, 126, 255]


    for i in test_index:
        root = '../original_data_whole/'
        point_i = np.loadtxt(root + str(i) + '/norm_3cut.txt',dtype=float)[:28,:2]
        with open(root+str(i)+'/label.txt', "r") as f:
            label = int(f.read().strip())
        if i in [269,271,272,275,276,277]:
            point_i = flip_coordinates(point_i)
        x.append(point_i)
        labels.append(label)
    x = torch.from_numpy(np.array(x,dtype = np.float32))  # 节点特征维度为 (13, 3)
    edge_index = np.loadtxt('../edge/edge3_index.txt')
    edge_index = torch.tensor(edge_index, dtype=torch.int)  # 边索引维度为 (2, 13)
    y = torch.tensor(labels, dtype=torch.float32)  # 图分类任务的类别标签

    return x,edge_index, y


if __name__=='__main__':
    out_data()
    out_testdata()