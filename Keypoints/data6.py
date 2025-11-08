import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import os
import json
import random

# Function to flip coordinates horizontally
def flip_coordinates(coords, max_x=512):
    return [(max_x - x, y) for x, y in coords]

def out_data():
    x = list()
    labels = list()
    arra = list()
    train_index = [275, 273, 268, 276, 202, 148, 247, 45, 199, 196, 121, 15, 269, 266, 36, 277, 16, 183, 231, 274, 272, 111, 120, 70, 271, 61, 38, 108, 218, 244, 62, 177, 198, 132, 256, 224, 195, 216, 125, 39, 60, 25, 168, 46, 30, 27, 210, 207, 35, 225, 37, 182, 101, 173, 72, 200, 151, 71, 44, 51, 126, 20, 252, 58, 136, 57, 213, 34, 144, 154, 253, 31, 243, 166, 53, 233, 23, 33, 129, 175, 261, 157, 140, 172, 176, 116, 43, 219, 130, 262, 230, 105, 258, 123, 227, 226, 63, 192, 19, 186, 239, 29, 217, 40, 201, 26, 163, 237, 242, 265, 147, 251, 211, 42, 205]

    for i in train_index:
        root = '../original_data_whole/'
        point_i = np.loadtxt(root + str(i) + '/norm_6cut.txt',dtype=float)[:10,:2]
        #ratio = random.randint(0,1)
        if i in [269,271,272,275,276,277]:
            point_i = flip_coordinates(point_i)
        with open(root+str(i)+'/label.txt', "r") as f:
            label = int(f.read().strip())
        x.append(point_i)
        labels.append(label)
    x = torch.from_numpy(np.array(x,dtype = np.float32))  # 节点特征维度为 (10, 2)
    edge_index = np.loadtxt('../edge/edge6_index.txt')
    edge_index = torch.tensor(edge_index, dtype=torch.int)  # 边索引维度为 (2, 13)
    y = torch.tensor(labels, dtype=torch.float32)  # 图分类任务的类别标签

    return x,edge_index, y

def out_testdata():
    x = list()
    labels = list()
    test_index = [99, 68, 135, 104, 41, 267, 203, 270, 110, 18, 54, 28, 191, 260, 263, 264, 139, 14, 153, 155, 158, 32, 162, 164, 165, 167, 169, 170, 47, 48, 178, 64, 65, 67, 69, 212, 215, 220, 221, 223, 100, 229, 102, 232, 107, 109, 238, 240, 241, 114, 245, 246, 248, 249, 122, 254, 255]

    for i in test_index:
        root = '../original_data_whole/'
        point_i = np.loadtxt(root + str(i) + '/norm_6cut.txt',dtype=float)[:10,:2]
        if i in [269,271,272,275,276,277]:
            point_i = flip_coordinates(point_i)
        with open(root+str(i)+'/label.txt', "r") as f:
            label = int(f.read().strip())
        x.append(point_i)
        labels.append(label)
    x = torch.from_numpy(np.array(x,dtype = np.float32))  # 节点特征维度为 (13, 3)
    edge_index = np.loadtxt('../edge/edge6_index.txt')
    edge_index = torch.tensor(edge_index, dtype=torch.int)  # 边索引维度为 (2, 13)
    y = torch.tensor(labels, dtype=torch.float32)  # 图分类任务的类别标签

    return x,edge_index, y


if __name__=='__main__':
    out_data()
    out_testdata()