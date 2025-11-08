import os
import torch
import time
import cv2
import numpy as np

from torch.autograd import Variable
from Laryngoscope.model import PUCLNet
from Laryngoscope.data import Data

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 270
num_worker = 0
root= "../original_data_whole/"


def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例
    ax = plt.subplot(111)       # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # if label[i]==0:
        #     plt.plot(data[i, 0], data[i, 1], 'o', color='red')
        # elif label[i]==1:
        #     plt.plot(data[i, 0], data[i, 1], 'd', color='blue')
        # 在图中为每个数据点画出标签
        if label[i] == 0:
            plt.text(data[i, 0], data[i, 1], str(i), color='red',
                 fontdict={'weight': 'bold', 'size': 7})
        elif label[i] == 1:
            plt.text(data[i, 0], data[i, 1], str(i), color='blue',
                 fontdict={'weight': 'bold', 'size': 7})

    plt.xticks()        # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig

if __name__=='__main__':

    dataset = Data(root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
    )
    Model = PUCLNet()#.cuda()
    path_model = "Larycheck/300.pth"
    Model.load_state_dict(torch.load(path_model,map_location='cpu'))

    Model.eval()
    for batch_i, (img, label,img_id) in enumerate(dataloader):
        batches_done = batch_i + 1

        img = Variable(img.float())#.cuda()
        label = label.float().detach().numpy()

        feature, cls=Model(img)

        pos_feature = list()
        neg_feature = list()

        label_pos = list()
        label_neg = list()

        feature  = feature.detach().numpy()
        for i in range(cls.shape[0]):
            #print(i,cls[i],label[i],img_id[i])
            if cls[i]>0.5 and label[i]>0.5:
                pos_feature.append(feature[i,:])
                label_pos.append(label[i])
            if cls[i]<0.5 and label[i]<0.5:
                neg_feature.append(feature[i,:])
                label_neg.append(label[i])
        pos_feature = np.array(pos_feature)
        neg_feature = np.array(neg_feature)
        label_pos = np.array(label_pos)
        label_neg = np.array(label_neg)

        feature1 = np.concatenate((pos_feature,neg_feature),axis=0)
        label1 = np.concatenate((label_pos,label_neg),axis=0)

        # pos = np.where(label>0.5)
        # neg = np.where(label<0.5)
        # pos_feature = feature[pos]
        # neg_feature = feature[neg]


        pos_proto = torch.from_numpy(pos_feature.mean(axis=0))
        neg_proto = torch.from_numpy(neg_feature.mean(axis=0))

        print(pos_proto, neg_proto)
        with open("pos_proto.txt", 'w') as f:
            np.savetxt(f, pos_proto)
        with open("neg_proto.txt", 'w') as f:
            np.savetxt(f, neg_proto)


        ts = TSNE(perplexity=5,n_components=2, init='pca', random_state=0)
        # t-SNE降维
        result = ts.fit_transform(feature1)

        # #PCA降维
        # pca = PCA(n_components=4)
        # result = pca.fit_transform(feature)

        #调用函数，绘制图像
        fig = plot_embedding(result, label1, 't-SNE Embedding of digits')
        #fig = plot_embedding(result, [1,0], 't-SNE Embedding of digits')
        #显示图像
        plt.show()



