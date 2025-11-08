import numpy as np
import torch


def adjacency_matrix_to_edge_index(adjacency_matrix):
    edge_index = []
    num_nodes = adjacency_matrix.shape[0]


    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index


# 示例邻接矩阵
adjacency_matrix = np.loadtxt('../edge/edge_3.txt',dtype=int)

for i in range(adjacency_matrix.shape[0]):
    for j in range(adjacency_matrix.shape[0]):
        if adjacency_matrix[i][j] == 1:  # 如果存在单向边 (i, j)
            if adjacency_matrix[j][i] != 1:
                print(j+1,i+1)
            adjacency_matrix[j][i] = 1


# 将邻接矩阵转换为 edge_index
edge_index = adjacency_matrix_to_edge_index(adjacency_matrix)
edge_index = edge_index.numpy()
np.savetxt('../edge/edge3_index.txt',edge_index, fmt='%d')