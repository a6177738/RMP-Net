import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ProtoLoss(nn.Module):
    ''' Classic loss function for face recognition '''
    def __init__(self):
        super(ProtoLoss, self).__init__()

    def forward(self, predict, label, pos_proto, neg_proto):

        t = 0.2
        #注意索引方法
        pos_loc = torch.where(label > 0.5)[0]
        pos_loc, _ = torch.unique(pos_loc, sorted=True, return_counts=True)
        neg_loc = torch.where(label < 0.5)[0]
        neg_loc, _ = torch.unique(neg_loc, sorted=True, return_counts=True)
        pos_predict = predict[pos_loc]
        neg_predict = predict[neg_loc]

        if len(pos_predict) != 0:
            pos_denominator = torch.exp(torch.matmul(pos_predict, pos_proto)/t) + torch.exp(torch.matmul(pos_predict, neg_proto)/t) #注意加法不要在exp括号内进行，否则损失会让predict趋近于0
            numerator_pos = torch.exp(torch.matmul(pos_predict, pos_proto)/t)
            pos_loss = (-torch.log(numerator_pos / pos_denominator)).mean()
        else:
            pos_loss = 0
        neg_denominator  = torch.exp(torch.matmul(neg_predict,pos_proto)/t)+torch.exp(torch.matmul(neg_predict,neg_proto)/t)
        numerator_neg = torch.exp(torch.matmul(neg_predict,neg_proto)/t)

        neg_loss = -torch.log(numerator_neg/neg_denominator).mean()

        uncer_cls_loss = pos_loss+neg_loss
        return uncer_cls_loss




if __name__ == "__main__":
   predict = torch.zeros((5,12))

   predict[2] += 1
   pos = torch.ones(12)
   neg = torch.zeros(12)

   label = torch.zeros(5)
   label[1] +=1
   label[2] +=1
   Cls_loss = Few_Shot_Loss(predict,label)
   print(Cls_loss)



