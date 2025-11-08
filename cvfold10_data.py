import numpy as np
from torch.utils.data import Dataset
import torch
import os
import cv2
from aug import all as augmentation
import random
import json

def traintestindex(root):
    pos_index = list()
    neg_index = list()

    for i in range(14, 73):
        json_file_path7 = root + str(i) + '/7.json'
        if not os.path.exists(json_file_path7):
            continue
        with open(json_file_path7, 'r') as file:
            # 使用json.load()读取JSON文件内容
            data = json.load(file)['shapes']
        if data[0]['label'] == '02':
            continue
        with open(root + str(i) + "/label.txt", "r") as f:
            label = int(f.read().strip())
            if label == 1:
                pos_index.append(i)
            else:
                neg_index.append(i)
    for i in range(99, 278):
        json_file_path7 = root + str(i) + '/7.json'
        if not os.path.exists(json_file_path7):
            continue
        with open(json_file_path7, 'r') as file:
            # 使用json.load()读取JSON文件内容
            data = json.load(file)['shapes']
        if data[0]['label'] == '02':
            continue
        with open(root + str(i) + "/label.txt", "r") as f:
            label = int(f.read().strip())
            if label > 0.5:
                pos_index.append(i)
            else:
                neg_index.append(i)
    train_posindex = random.sample(pos_index, k=26)
    train_negindex = random.sample(neg_index, k=89)
    train_index = train_posindex + train_negindex

    test_posindex = list(set(pos_index) - set(train_posindex))
    test_negindex = list(set(neg_index) - set(train_negindex))
    test_index = test_posindex[:] + test_negindex

    return train_index,test_index

class train_Data(Dataset):
    def __init__(self, root, tindex):
        self.root = root
        self.img_ids= list()
        self.gt_ids = list()
        self.mode_index = tindex
        for index, i in enumerate(self.mode_index):
            #self.img_ids.append((root + str(i) + "/" + "3_cut.jpg"))
            self.img_ids.append((root + str(i) + "/" + "6_cut.jpg"))
            self.gt_ids.append((root+str(i)+"/label.txt"))

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        label_id = self.gt_ids[index]
        view = cv2.imread(img_id, cv2.IMREAD_COLOR)

        with open(label_id,"r") as f:
            label = int(f.read().strip())
        view = cv2.resize(view,(224,224))
        view = view.copy()
        view = augmentation(view)
        view = view.transpose(2,0,1)/255
        return view, label

    def __len__(self):
        return len(self.img_ids)

class test_Data(Dataset):
    def __init__(self, root, tindex):
        self.root = root
        self.img_ids = list()
        self.gt_ids = list()
        self.mode_index = tindex
        for index, i in enumerate(self.mode_index):
            #self.img_ids.append((root + str(i) + "/" + "3_cut.jpg"))
            self.img_ids.append((root + str(i) + "/" + "6_cut.jpg"))
            self.gt_ids.append((root+str(i)+"/label.txt"))

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        label_id = self.gt_ids[index]
        with open(label_id,"r") as f:
            label = int(f.read().strip())
        view = cv2.imread(img_id, cv2.IMREAD_COLOR)

        if view is None or view.size == 0:
            print(img_id)

        view = cv2.resize(view,(224,224))
        view = view.copy()
        view = view.transpose(2,0,1)/255
        return view,label

    def __len__(self):
        return len(self.img_ids)



