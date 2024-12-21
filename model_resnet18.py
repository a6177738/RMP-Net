import torch
from torch import nn
from model.prior import PriorNet
import torchvision.models as tm
import numpy as np
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.prior = tm.resnet18(pretrained=True)

        self.fc = nn.Linear(1000, 256)
        self.fc1 = nn.Linear(256, 1)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, img):
        x = self.prior(img)

        feature = self.fc(x)
        feature = torch.nn.functional.normalize(feature, dim=1)

        mu = torch.sigmoid(self.fc1(feature))
        theta = F.softplus(self.fc2(feature))

        return  feature, mu, theta
        #return mu


