import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
from loss import ProtoLoss
import os
import json
from Keypoints.test import test
from Keypoints.data3 import out_data,out_testdata

# 构建示例数据
x, edge_index, y  = out_data()
Max_Gmean = 0
#print(y.sum(),(1-y).sum())

x_test, edge_index_test, y_test= out_testdata()

#print(y_test.sum(),(1-y_test).sum())

# 构建图数据
data = Data(x=x, edge_index=edge_index, y=y)
print(x.shape)
test_data = Data(x=x_test, edge_index=edge_index_test, y=y_test)

pos_proto = np.loadtxt("../Laryngoscope/pos_proto.txt")
neg_proto = np.loadtxt("../Laryngoscope/neg_proto.txt")
pos_proto1 = torch.tensor(pos_proto).clone().detach().float()
neg_proto1 = torch.tensor(neg_proto).clone().detach().float()
#
# 定义图分类模型
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(2,32)
        #self.conv2 = GCNConv(32, 32)

        # self.fc0 = torch.nn.Linear(1792, 1000)
        self.fc1 = torch.nn.Linear(896, 256)
        self.fc2 = torch.nn.Linear(256, 2) # 输出类别为 1，不确定行为1


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        #x = self.conv2(x, edge_index)
        x = x.view(x.shape[0],896)
        x = self.fc1(x)
        x = F.normalize(x)
        x1 = self.fc2(x)
        mu,theta = torch.sigmoid(x1[:,0]).squeeze(),x1[:,1].squeeze()
        theta = F.softplus(theta)
        return x, mu, theta

# 初始化模型
model = GNN()
model.train()

# 定义损失函数和优化器
weight = data.y * 0.2 + 1
criterion = torch.nn.BCELoss(reduction='none')#(weight=weight, reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=0.0005)

# path_model = "../check/9990.pth"
# model.load_state_dict(torch.load(path_model, map_location='cpu'))

# 进行训练和测试
for epoch in range(0, 10000):
    optimizer.zero_grad()
    feature,mu, theta = model(data)

    # 找到 theta 最高的 5 个数据点的索引
    _, highest_theta_indices = torch.topk(theta,1)
    # 创建一个布尔掩码来选择非最高 theta 的元素
    mask = torch.ones(len(theta), dtype=torch.bool)
    mask[highest_theta_indices] = False
    # 从 mu 和 theta 中移除这些索引对应的元素
    # 使用布尔掩码过滤 mu、theta 和 y_true
    feature_filter= feature[mask]
    mu_filter = mu[mask]
    theta_filter = theta[mask]
    y_filter = data.y[mask]

    Proto_loss = ProtoLoss()
    proto_loss = Proto_loss(feature_filter, y_filter, pos_proto1, neg_proto1)
    rg_loss = (1.5*criterion(mu_filter,y_filter)/theta_filter+torch.log(theta_filter+1)).mean()
    loss = rg_loss+proto_loss
    loss.backward()
    optimizer.step()

    feature_test, mu_test,theta_test = model(test_data)

    if epoch%5 == 0:
        print(f'Epoch: {epoch}', loss)
        max_Gmean = test(epoch,mu_test, test_data.y,Max_Gmean)
        if Max_Gmean<max_Gmean:
            Max_Gmean = max_Gmean
            parapth = "../check/" + str(epoch) + ".pth"
            torch.save(model.state_dict(), parapth)
print(Max_Gmean)




