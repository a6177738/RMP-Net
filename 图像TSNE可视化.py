import os
import torch
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from model.model_resnet18 import Net
from data import train_Data, test_Data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 200
num_worker = 0
root = "original_data_whole/"


def sample_and_average(features, labels, sample_size=3):
    zero_indices = np.where(labels == 0)[0]
    one_indices = np.where(labels == 1)[0]

    np.random.shuffle(zero_indices)
    sampled_zero_indices = [zero_indices[i:i + sample_size] for i in range(0, len(zero_indices), sample_size)]

    new_features = []
    new_labels = []

    for group in sampled_zero_indices:
        if len(group) == sample_size:
            new_features.append(np.mean(features[group], axis=0))
            new_labels.append(0)

    new_features.extend(features[one_indices])
    new_labels.extend(labels[one_indices])

    return np.array(new_features), np.array(new_labels)


if __name__ == '__main__':
    # train_index, test_index = traintestindex(root)
    train_index = [70, 202, 267, 203, 269, 271, 16, 275, 277, 54, 183, 120, 61, 256, 129, 263, 136, 144, 23, 151, 27, 29, 157, 31, 35, 163, 164, 165, 39, 167, 169, 172, 177, 51, 53, 57, 62, 63, 64, 65, 71, 72, 215, 221, 223, 224, 227, 100, 229, 105, 233, 244, 249, 122, 123, 254, 255]

    dataset = train_Data(root, train_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
    )
    Net1 = Net()
    Net1.eval()
    parapth = "check_img/" + str(262) + "proto7.pth"
    Net1.load_state_dict(torch.load(parapth, map_location='cpu'))


    for batch_i, (img, label) in enumerate(dataloader):
        batches_done = batch_i + 1
        img = Variable(img.float())
        feature, mu, theta = Net1(img)

        feature = feature.detach().numpy()
        label = label.detach().numpy()

        # # 对类别0的样本每三个进行融合，保留类别1的样本不变
        # feature, label = sample_and_average(feature, label, sample_size=3)

        # # 标准化特征
        # scaler = StandardScaler()
        # feature_scaled = scaler.fit_transform(feature)

        # 添加图例标签
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', markersize=10, label='Easy'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Difficult')
        ]

        # 设置颜色
        colors = np.array(['navy' if lab == 0 else 'orange' for lab in label])

        # 应用TSNE
        tsne = TSNE(n_components=2, random_state=38)#, perplexity=30, learning_rate=200)
        X_tsne = tsne.fit_transform(feature)

        # 可视化
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='viridis', alpha=0.7)
        #plt.colorbar(scatter)
        plt.title('TSNE Visualization')
        # plt.xlabel('TSNE Component 1')
        # plt.ylabel('TSNE Component 2')
        plt.savefig('/Users/lixiaofan/Desktop/原型不确定性困难气道评估/protoairway/可视化png/filterproto1.png')
        plt.show()
