from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from Keypoints.test import test
from torch_geometric.data import Data
from Keypoints.data3 import out_data,out_testdata

from sklearn.preprocessing import StandardScaler

Max_Gmean = 0
# 构建示例数据
x_train, edge_index, y_train  = out_data()
print(x_train.shape)
x_test, edge_index_test, y_test= out_testdata()

x_train = x_train.numpy()
y_train = y_train.numpy()
x_test = x_test.numpy()
y_test = y_test.numpy()

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# 定义KNN模型，假设k=5
knn = KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn.fit(x_train_scaled, y_train)

# 进行预测
y_pred = knn.predict(x_test_scaled)

Max_Gmean = test(0, y_pred, y_test, Max_Gmean)
