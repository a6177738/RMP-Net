import os
import torch
import time
import cv2

from torch.autograd import Variable
from model_laryngoscope import PUCLNet
from Laryngoscope.data import Data
from loss import Few_Shot_Loss
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 250
num_worker = 0
lr = 8e-3
momentum = 0.9
weight_decay = 0.0005
epochs  = 301
root= "../original_data_whole/"

if __name__=='__main__':

    dataset = Data(root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=T,
        num_workers=num_worker,
    )
    Model = PUCLNet()#.cuda()

    optimizer = torch.optim.SGD(Model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    path_model = "Larycheck/280.pth"
    Model.load_state_dict(torch.load(path_model,map_location='cpu'))
    for epoch in range(281,epochs):
        Model.train()
        for batch_i, (img, label) in enumerate(dataloader):
            batches_done = batch_i + 1

            img = Variable(img.float())#.cuda()
            label = label.float().squeeze()

            feature,cls=Model(img)

            weight = label*0.2+1
            Loss = torch.nn.BCELoss(weight=weight)
            cls_loss = Loss(cls,label)
            proto_loss = Few_Shot_Loss(feature,label)
            loss =  cls_loss+proto_loss
            loss.backward()

            if batches_done % 2:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            print("e:", epoch, " b_i:", batch_i, " cls_loss:", str(cls_loss)[7:14],str(proto_loss)[7:14])
        if epoch%10==0:
            parapth = "Larycheck/" + str(epoch) + ".pth"
            torch.save(Model.state_dict(), parapth)
