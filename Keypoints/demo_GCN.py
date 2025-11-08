import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from Keypoints.data3 import out_data, out_testdata
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# 构建示例数据
x, edge_index, y = out_data()
# 构建图数据
data = Data(x=x, edge_index=edge_index, y=y)

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(2, 32)
        self.fc1 = torch.nn.Linear(896, 256)
        self.fc2 = torch.nn.Linear(256, 2)  # 输出类别为2
        self.activations = None  # 初始化为None
        self.gradients = None  # 初始化为None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        self.activations = x  # 存储激活值

        # Register hook to save gradients
        self.activations.register_hook(self.save_gradients)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.normalize(x)
        x1 = self.fc2(x)
        mu, theta = torch.sigmoid(x1[:, 0]).squeeze(), x1[:, 1].squeeze()
        return mu, theta

    def save_gradients(self, grad):
        self.gradients = grad

print(data.x.shape)
# 加载模型参数
model = GNN()
model.load_state_dict(torch.load('/Users/lixiaofan/Desktop/原型不确定性困难气道评估/protoairway/check/550.pth'))
model.eval()

def compute_gradcam(model, data):
    data.x.requires_grad = True

    output, theta = model(data)
    loss = output.sum()  # 针对输出进行求和以计算梯度
    loss.backward(retain_graph=True)

    # 提取卷积层的梯度和激活值
    gradients = model.gradients  # 获取卷积层的梯度
    activations = model.activations.detach()  # 获取卷积层的激活值

    # 计算Grad-CAM权重
    weights = torch.mean(gradients, dim=1, keepdim=True)  # 计算每个特征的权重
    cam = torch.sum(weights * activations, dim=2)  # 按特征加权求和

    print(output.detach().numpy())
    print(theta.detach().numpy())

    return cam.numpy(), activations.numpy()

# 获取样本数
num_samples = data.x.shape[0]  # 57个样本

# 处理每个样本并保存可视化结果
for i in range(num_samples):
    # 获取当前样本的节点坐标
    sample_x = data.x[i].unsqueeze(dim=0)  # 使用unsqueeze调整维度

    # 创建单个样本的数据对象
    sample_data = Data(x=sample_x, edge_index=data.edge_index)

    # 计算Grad-CAM
    cam, activations = compute_gradcam(model, sample_data)

    # 调试信息
    print(f"Sample {i}: CAM shape = {cam.shape}, Node count = {sample_data.x.shape[1]}")

    # 确保 cam 的大小与节点数量一致
    cam = cam.flatten()

    # 创建NetworkX图并添加边
    G = nx.Graph()
    for node in range(sample_data.x.shape[1]):
        G.add_node(node)  # 添加所有节点
    for j in range(len(sample_data.edge_index[0])):
        u = sample_data.edge_index[0][j].item()
        v = sample_data.edge_index[1][j].item()
        # 计算边的权重
        edge_weight = (activations[0][u] * activations[0][v]).sum()
        G.add_edge(u, v, weight=edge_weight)

    # 确保 pos 包含所有节点的坐标
    pos = {k: (sample_x[0][k][0].item(), sample_x[0][k][1].item()) for k in range(sample_x.shape[1])}

    # 确保 pos 和 cam 的大小一致
    if len(pos) != len(cam):
        raise ValueError(f"Size mismatch: pos has {len(pos)} elements but cam has {len(cam)} elements")

    # 为没有边的节点添加默认颜色值
    node_colors = np.zeros(len(pos))
    for k, color in zip(pos.keys(), cam):
        node_colors[k] = color

    edges = G.edges(data=True)
    edge_weights = [edge[2]['weight'] for edge in edges]

    plt.figure(figsize=(8, 6))

    # 画节点
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, cmap=plt.cm.Reds)

    # 手动绘制边
    for (u, v, d), weight in zip(G.edges(data=True), edge_weights):
        x_coords = [pos[u][0], pos[v][0]]
        y_coords = [pos[u][1], pos[v][1]]
        # 使用红色渐变而不是蓝色渐变
        plt.plot(x_coords, y_coords, color=plt.cm.Reds(weight / max(edge_weights)),
                 linewidth=2 + 5 * (weight / max(edge_weights)))

    # 画标签
    nx.draw_networkx_labels(G, pos)

    # 反转y轴
    plt.gca().invert_yaxis()

    # 添加节点颜色条
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.Reds,
                                     norm=plt.Normalize(vmin=node_colors.min(), vmax=node_colors.max()))
    sm_nodes.set_array([])
    plt.colorbar(sm_nodes, label="Node CAM")

    # 添加边颜色条
    sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                                     norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm_edges.set_array([])
    plt.colorbar(sm_edges, label="Edge CAM")

    # 保存图像
    plt.savefig(f'train_cam3/{str(i+99)}.png')
    plt.close()
