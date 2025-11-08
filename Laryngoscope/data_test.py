import numpy as np
from torch.utils.data import Dataset
import torch
import os
import cv2
from aug import all as augmentation
import random
import json

class Data(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_ids = list()
        self.gt_ids = list()

        for i in range(13,266):
            json_file_path7 = root + str(i) + '/7.json'
            if not os.path.exists(json_file_path7):
                continue
            with open(json_file_path7, 'r') as file:
                # 使用json.load()读取JSON文件内容
                data = json.load(file)['shapes']
            if data[0]['label'] != '02':
                continue
            self.img_ids.append((root + str(i) + "/" + "7.png"))
            self.gt_ids.append((root+str(i)+"/label.txt"))



    def __getitem__(self, index):
        img_id = self.img_ids[index]
        label_id = self.gt_ids[index]

        with open(label_id,"r") as f:
            label = int(f.read().strip())

        view = cv2.imread(img_id, cv2.IMREAD_COLOR)
        view = cv2.resize(view,(224,224))
        view = augmentation(view)
        view = view.transpose(2,0,1)/255

        return view,label
    def __len__(self):
        return len(self.img_ids)
