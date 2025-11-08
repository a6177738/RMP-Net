import os
import torch
import time
import numpy as np
from ROC import roc
from sklearn.metrics import matthews_corrcoef as mcc

def create_file(filename):
    if not os.path.exists(filename):
        open(filename, 'w').close()
        print("文件创建成功")
    else:
        print("文件已存在")


def test(pth, Cls, Label,max_Gmeans):
        Max_Gmeans = max_Gmeans

        clss  = Cls.detach().numpy().squeeze()
        labels = Label.detach().numpy().squeeze()
        fpr, tpr, auc = roc(labels,clss)
        k = 0.001
        ACC = 0
        Gmean  = 0
        z = 1
        PRE = 0
        SEN =0
        SPE = 0
        MCC = 0
        F1 = 0
        p = 0
        while(k<1):
            cls1 =  np.where(clss>k,1,0)
            TP = (cls1*labels).sum()
            TN = ((1-labels)*(1-cls1)).sum()
            FP = (cls1*(1-labels)).sum()
            FN = ((1-cls1)*labels).sum()

            Pre = TP/(TP+FP+0.00001)

            Sen = TP / (TP+FN+0.00001)
            Spe = TN / (TN+FP)

            Mcc = (TP*TN-FP*FN)/(pow(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),0.5)+0.00001)
            #Mcc = mcc(labels,cls1)

            Acc = (TP+TN)/(TP+TN+FP+FN)

            f1 = (2*Pre*Sen)/(Pre+Sen+0.00001)

            if pow(Sen*Spe,0.5)>=Gmean:
                Gmean = pow(Sen*Spe,0.5)
                PRE = Pre
                F1 = f1
                MCC = Mcc
                ACC = Acc
                SEN = Sen
                SPE = Spe
                p = k
            k = k+0.0001

        print("epoch:", pth," t:", p, " Gmean:", Gmean, " acc:", ACC, " Precision:", PRE, ' F1:', F1,
              ' Sen:', SEN, ' Spe:', SPE, ' Mcc:', MCC, ' AUC:', auc)
        if Gmean>max_Gmeans:
            # with open('../Keypoints/graph_AUC/RF1_fpr.txt', 'w') as pf:
            #     # 使用 numpy.savetxt 写入文件
            #     np.savetxt(pf, fpr, fmt='%f')
            # with open('../Keypoints/graph_AUC/RF1_tpr.txt', 'w') as nf:
            #     # 使用 numpy.savetxt 写入文件
            #     np.savetxt(nf, tpr, fmt='%f')
            Max_Gmeans = Gmean
        return Max_Gmeans
