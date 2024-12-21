import torch

from torch import nn
from Laryngoscope.resnet import resnet11
import torchvision.models as tm

class PUCLNet(nn.Module):
    def __init__(self):
        super(PUCLNet,self).__init__()
        self.resnet = tm.resnet18()
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(1000,256)
        self.linear1 = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
        for param in self.resnet.parameters():
            if len(param.size()) > 1:  # 只对权重进行初始化
                nn.init.xavier_uniform_(param)

    def forward(self, img):
        feature= self.resnet(img)
        feature = self.linear(feature)
        feature = torch.nn.functional.normalize(feature,dim=1)
        cls = self.linear1(feature).squeeze()
        cls = torch.sigmoid(cls)
        return feature, cls
