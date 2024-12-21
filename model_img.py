import torch
from torch import nn
from model.prior import PriorNet
import torchvision.models as tm
import numpy as np
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, mode = 'train'):
        super(Net, self).__init__()

        self.prior = tm.resnet18(pretrained=False)

        self.fc = nn.Linear(1000, 256)
        self.fc1 = nn.Linear(256, 1)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, img):
        x = self.prior(img)
        feature = self.fc(x)
        feature = torch.nn.functional.normalize(feature, dim=1).squeeze(dim=1)

        mu = torch.sigmoid(self.fc1(feature)).squeeze(dim=1)
        theta = F.softplus(self.fc2(feature)).squeeze(dim=1)

        return  feature, mu, theta

