from skimage import io
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np
import os

np.set_printoptions(threshold=10000)

for i in range(222,240):
    # 假设json_file_path是您的JSON文件的路径
    root = '../original_data_whole/'
    json_file_path = root+str(i)+'/3.json'
    if not os.path.exists(json_file_path):
        continue
    img = cv2.imread(root+str(i)+'/3_cut.jpg')
    img = cv2.resize(img,(512,512))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with open(root + str(i) + '/norm_3cut.txt', 'r') as f:
        points = np.loadtxt(f,dtype=int)
        points = points[:,:2]
        #tri = Delaunay(points)

        # 确定顶点数量
        num_vertices = points.shape[0]

        # 创建邻接矩阵（初始全为0）
        #adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

        # 遍历每个三角形，确定三角形的边，更新邻接矩阵
        # for simplex in tri.simplices:
        #     # 对于每条边，更新邻接矩阵中对应位置为1
        #     for i in range(3):
        #         p1, p2 = simplex[i], simplex[(i + 1) % 3]
        #         adjacency_matrix[p1, p2] = 1
        #         adjacency_matrix[p2, p1] = 1  # 因为是无向图，所以对称地设置对应位置为1

        adjacency_matrix = np.loadtxt('../edge/edge_3.txt',dtype=int)

        adjacency_matrix = np.array(adjacency_matrix)

        #print(adjacency_matrix.shape)
        for k in range(points.shape[0]):
            for z in range(k + 1, points.shape[0]):
                if adjacency_matrix[k, z] == 1:
                    plt.plot([points[k, 0], points[z, 0]], [points[k, 1], points[z, 1]], 'k-')

        # 打印邻接矩阵
        #print(adjacency_matrix)

        plt.gca().invert_yaxis()
        #plt.imshow(img)
        #plt.triplot(t_points[:, 0], t_points[:, 1], tri.simplices)
        plt.plot(points[:, 0], points[:, 1], 'o')
        plt.title('Delaunay Triangulation')
        plt.xlabel('X')
        plt.ylabel('Y')
        #plt.show()
        print(i)
        plt.savefig('/result/'+str(i)+'_3.png')
        plt.clf()

# plt.imshow(image_RGB)
# plt.axis("off")
# plt.show()
