import os
import torch
import time
import cv2
import numpy as np

from torch.autograd import Variable
import torch.nn.functional
from model.model_resnet18 import Net
from data import test_Data
from test  import test
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def transnumpy(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        data = file.read()
    # 移除换行符，将整个内容分成每个数组的字符串
    arrays = data.strip().split('\n')
    # 去除方括号和多余的空格，并将每个数组字符串转换为列表
    arrays = [array.replace('[', '').replace(']', '').strip().split(',') for array in arrays]
    # 转换为 numpy 数组
    arrays = [np.array(array, dtype=float) for array in arrays]
    # 重新调整形状，合并成一个二维数组
    mu, theta  = arrays[0],arrays[1]
    return mu,theta
batch_size = 64
num_worker = 0
root= "original_data_whole/"

def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())+0.0001

def mean_fusion(mu1, theta1, mu2, theta2, alpha=2):
    # 确保输入为 torch 张量
    mu1 = torch.tensor(mu1, dtype=torch.float32)
    mu2 = torch.tensor(mu2, dtype=torch.float32)
    theta1 = torch.tensor(theta1, dtype=torch.float32)
    theta2 = torch.tensor(theta2, dtype=torch.float32)
    # 计算加权均值
    fused_mu = (alpha * mu1 + mu2) / (alpha + 1)
    # 计算融合后的不确定性（方差），这里简单使用均值的不确定性
    fused_theta = torch.sqrt((theta1**2 + theta2**2) / 2)

    return fused_mu, fused_theta

def fuse_predictions(mu1, theta1, mu2, theta2):
    alpha = 0.2

    #mu1 = normalize(mu1)
    #mu2 = normalize(mu2)

    # 计算融合后的均值
    fused_mu = (alpha * mu1 / theta1 ** 2 + mu2 / theta2 ** 2) / (alpha / theta1 ** 2 + 1/ theta2 ** 2)

    # 计算融合后的不确定性（方差）
    fused_theta_squared = 1 / (alpha * (1 / theta1 ** 2) + 1 / theta2 ** 2)
    fused_theta = torch.sqrt(fused_theta_squared)

    return fused_mu, fused_theta

if __name__=='__main__':
    # 创建一个包含57个数的张量
    label_test = torch.zeros(57)
    # 将前13个数设置为1
    label_test[:13] = 1
    mu1,theta1 = transnumpy('单模态输出/img2.txt')
    mu2,theta2 = transnumpy('单模态输出/graph7.txt')

    print(mu1,theta1,mu2,theta2)


    # 确保输入为 torch 张量
    mu1 = torch.tensor(mu1, dtype=torch.float32)
    theta1 = torch.tensor(theta1, dtype=torch.float32)
    mu2 = torch.tensor(mu2, dtype=torch.float32)
    theta2 = torch.tensor(theta2, dtype=torch.float32)

    ratio = theta1.mean()/theta2.mean()
    theta1_norm = theta1
    theta2_norm = theta2

    # 计算融合结果
    fused_mu, fused_theta = fuse_predictions(mu1, theta1_norm, mu2, theta2_norm)
    print(ratio,fused_mu,fused_theta)
    #fused_mu, fused_theta = mean_fusion(mu1, theta1, mu2, theta2)
    test(1, fused_mu, label_test, 0)
