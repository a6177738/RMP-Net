import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from Keypoints.data6 import out_data, out_testdata
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np

x, edge_index, y = out_testdata()
data = Data(x=x, edge_index=edge_index, y=y)

# 定义图分类模型
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(2, 32)
        self.fc1 = torch.nn.Linear(320, 256)
        self.fc2 = torch.nn.Linear(256, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = x.view(x.shape[0], 320)
        x = self.fc1(x)
        x = F.normalize(x)
        x1 = self.fc2(x)
        mu, theta = torch.sigmoid(x1[:, 0]).squeeze(), x1[:, 1].squeeze()
        theta = F.softplus(theta)
        return x, mu, theta

# 初始化模型
model = GNN()
model.train()

path_model = "/Users/lixiaofan/Desktop/原型不确定性困难气道评估/protoairway/check_graph/115_pc8.pth"
model.load_state_dict(torch.load(path_model, map_location='cpu'))
feature, mu, theta = model(data)
feature = feature.detach().numpy()

# 应用TSNE
tsne = TSNE(n_components=2,random_state=2)#,perplexity=2)
X_tsne = tsne.fit_transform(feature)

# 获取类别标签
labels = data.y.detach().numpy()

# 对类别较多的类别进行抽样
class_0_indices = np.where(labels == 0)[0]
class_1_indices = np.where(labels == 1)[0]

#按比例抽样，确保比例一致
num_class_0 = len(class_0_indices)
num_class_1 = len(class_1_indices)
sample_ratio = num_class_1 / num_class_0

np.random.shuffle(class_0_indices)
sampled_class_0_indices = class_0_indices[:int(sample_ratio * num_class_0)]

balanced_indices = np.concatenate((sampled_class_0_indices, class_1_indices))
X_balanced = X_tsne[balanced_indices]
y_balanced = labels[balanced_indices]

# 设置颜色
colors = np.array(['navy' if label == 0 else 'orange' for label in labels])

# 绘制TSNE和SVM分割线
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.7)

# 添加图例标签
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', markersize=10, label='Easy'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Difficult')
]
plt.legend(handles=handles, loc='upper right')

plt.title('Basic+Proto+Uncertainty')

plt.savefig('/Users/lixiaofan/Desktop/原型不确定性困难气道评估/protoairway/可视化png/聚类/graph_pc8.png')
plt.show()
