import os
import torch
import time
import cv2
import numpy as np

from torch.autograd import Variable
import torch.nn.functional
from model.model_proto import Net
from cvfold10_data import train_Data,test_Data,traintestindex
from loss import ProtoLoss,prototype_nce_loss
from cvfold10_test  import test
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 64
num_worker = 4
lr = 0.004
momentum = 0.9
weight_decay = 0.0005
epochs  = 301
max_Gmeans = 0
root= "original_data_whole/"
pos_proto = np.loadtxt("Laryngoscope/pos_proto.txt")
neg_proto = np.loadtxt("Laryngoscope/neg_proto.txt")

if __name__=='__main__':
   # train_index, test_index = traintestindex(root)
    train_index = [271, 202, 266, 16, 15, 99, 247, 273, 276, 111, 135, 70, 277, 183, 270, 269, 110, 18, 196, 45, 121, 275, 54, 36, 267,
 120, 211, 265, 240, 198, 39, 20, 221, 260, 30, 245, 105, 220, 32, 154, 136, 35, 48, 38, 130, 175, 65, 157, 263, 262,
 229, 177, 158, 246, 63, 244, 64, 261, 60, 27, 226, 165, 213, 167, 224, 253, 258, 237, 178, 44, 140, 223, 200, 29, 51,
 170, 53, 37, 14, 218, 46, 57, 219, 195, 31, 122, 256, 169, 100, 34, 19, 69, 163, 212, 186, 172, 101, 264, 58, 232, 227,
 205, 151, 72, 251, 47, 243, 230, 217, 25, 114, 108, 67, 248, 242]

#train_index = train_index#+train_index[:26]
    test_index = [68, 199, 231, 41, 104, 203, 268, 272, 274, 148, 28, 61, 191, 129, 132, 139, 144, 147, 23, 153, 26, 155, 33, 162, 164,
 166, 40, 168, 42, 43, 173, 176, 182, 62, 192, 71, 201, 207, 210, 215, 216, 225, 102, 233, 107, 109, 238, 239, 241, 116,
 254, 249, 123, 252, 125, 126, 255]

    print(train_index,test_index)

    dataset = train_Data(root, train_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
    )

    dataset1 = test_Data(root, test_index)
    dataloader1 = torch.utils.data.DataLoader(
        dataset1,
        batch_size=30,
        shuffle=False,
        num_workers=num_worker,
    )

    Net1 = Net().cuda()
   # Net2  = Net('test').cuda()
    optimizer = torch.optim.SGD(Net1.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    highest_theta_indices = [0]

    for epoch in range(0,epochs):
        Net1.train()
        for batch_i, (img, label) in enumerate(dataloader):
            batches_done = batch_i + 1

            img = Variable(img.float()).cuda()
            label = label.detach().float().cuda()

            feature, mu, theta = Net1(img)
            mu = mu.squeeze()
            theta = theta.squeeze()
            # 找到 theta 最高的 3 个数据点的索引
            mask = torch.ones(len(theta), dtype=torch.bool)
            if epoch == 100:
                _, highest_theta_indices = torch.topk(theta, 3)
            #
            # # 创建一个布尔掩码来选择非最高 theta 的元素
            if epoch > 100:
                mask[highest_theta_indices] = False
            # 从 mu 和 theta 中移除这些索引对应的元素
            # 使用布尔掩码过滤 mu、theta 和 y_true
            feature_filter = feature[mask]
            mu_filter = mu[mask].squeeze()
            theta_filter = theta[mask]
            label_filter = label[mask]

            # pos_proto1 = torch.tensor(pos_proto).clone().detach().float().cuda()
            # neg_proto1 = torch.tensor(neg_proto).clone().detach().float().cuda()
            #
            # Loss = ProtoLoss()
            # proto_loss = prototype_nce_loss(feature_filter,label_filter,pos_proto1,neg_proto1)
            #kl_loss = -0.5*(1+torch.log(fea_theta*fea_theta)-fea_mu*fea_mu-fea_theta*fea_theta).mean()

            weight = label+0.2
            BCE_Loss = torch.nn.BCELoss()
            #cls_loss = (mu-label).pow(2)
            cls_loss  = BCE_Loss(mu_filter,label_filter)

            #rg_loss = (cls_loss*1.5/theta_filter+torch.log(theta_filter+0.5)).mean()
            loss = cls_loss#+kl_loss+proto_loss
            #loss = rg_loss+proto_loss#+kl_loss

            loss.backward()

            if batches_done % 1:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.empty_cache()

            #print("e:", epoch, " b_i:", batch_i, " rg_loss:", str(cls_loss.mean())[7:14], str(rg_loss)[7:14])
            print("e:", epoch, " b_i:", batch_i, " cls_loss:", str(cls_loss)[7:14])
        if epoch %2 == 0:
            # parapth = "check/" + str(epoch) + "resnet181.pth"
            # torch.save(Net1.state_dict(),parapth)
            #Net2.load_state_dict(torch.load(parapth))
            #Net2.eval()
            ccc = 0
            torch.cuda.empty_cache()
            Cls = list()
            Label = list()
            for batch_itest, (img_test, label_test) in enumerate(dataloader1):
                torch.cuda.empty_cache()
                img_test = Variable(img_test.float()).cuda()
                label_test = label_test.squeeze().detach().numpy()
                feature_test, mu_test, theta_test = Net1(img_test)
                mu1_test = mu_test.squeeze().cpu().detach().numpy()
                theta1_test = theta_test.squeeze().cpu().detach().numpy()
                Cls.append(mu1_test)
                Label.append(label_test)
            Cls_test = np.concatenate((Cls[0],Cls[1]),axis=0)
            Label_test = np.concatenate((Label[0],Label[1]),axis=0)
            max_Gmeans = test(epoch, Cls_test, Label_test, max_Gmeans)
