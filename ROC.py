from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
# 假设y_true是真实标签，y_score是预测得分或概率
# 计算ROC曲线
def roc(y_true,y_score):

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    #print(fpr,tpr,thresholds)

    # 计算AUC值
    roc_auc = auc(fpr, tpr)
    return fpr,tpr,roc_auc


    #print(roc_auc)
