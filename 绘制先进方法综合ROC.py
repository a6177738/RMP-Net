import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy import interpolate

# 定义统一的FPR范围
mean_fpr = np.linspace(0, 1, 50)


# 读取和处理数据的函数
def read_and_interpolate(fpr_files, tpr_files, mean_fpr):
    fprs = [np.loadtxt(file) for file in fpr_files]
    tprs = [np.loadtxt(file) for file in tpr_files]
    interp_tprs = []
    aucs = []
    for fpr, tpr in zip(fprs, tprs):
        interpolator = interpolate.interp1d(fpr, tpr, kind='linear')
        interpolated_tpr = interpolator(mean_fpr)
        interp_tprs.append(interpolated_tpr)
        aucs.append(auc(fpr, tpr))
    return np.array(interp_tprs), np.array(aucs)


# 文件名列表
models = {
    "Lin et al.": (['AUC/pfld{}_fpr.txt'.format(i) for i in range(6, 11)],
                   ['AUC/pfld{}_tpr.txt'.format(i) for i in range(6, 11)]),
    "Hayasaka et al.": (['AUC/hayasaka{}_fpr.txt'.format(i) for i in range(6, 11)],
                        ['AUC/hayasaka{}_tpr.txt'.format(i) for i in range(6, 11)]),
    "Wang et al.": (['AUC/mixup{}_fpr.txt'.format(i) for i in range(1, 6)],
                    ['AUC/mixup{}_tpr.txt'.format(i) for i in range(1, 6)]),
    "García-García et al.": (['复现/tprfpr/fpr_best{}.txt'.format(i) for i in range(1, 6)],
                    ['复现/tprfpr/tpr_best{}.txt'.format(i) for i in range(1, 6)]),
    "Li et al.": (['AUC/mce{}_fpr.txt'.format(i) for i in range(1, 6)],
                   ['AUC/mce{}_tpr.txt'.format(i) for i in range(1, 6)]),
    "Xia et al.": (['AUC/xia{}_fpr.txt'.format(i) for i in range(1, 6)],
                   ['AUC/xia{}_tpr.txt'.format(i) for i in range(1, 6)]),
    "RMP-Net": (['AUC/fuse{}_fpr.txt'.format(i) for i in range(1, 6)],
                 ['AUC/fuse{}_tpr.txt'.format(i) for i in range(1, 6)])
}

# 计算各模型的平均和标准差
results = {}
for model_name, (fpr_files, tpr_files) in models.items():
    interp_tprs, aucs = read_and_interpolate(fpr_files, tpr_files, mean_fpr)
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0  # 确保最后一个点是1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    results[model_name] = {
        "mean_tpr": mean_tpr,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "interp_tprs": interp_tprs
    }

    # 打印均值和方差
    print(f'Model: {model_name}, Mean AUC: {mean_auc:.4f}, Std AUC: {std_auc:.4f}')

# 绘制平均ROC曲线
plt.figure()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf','#8c564b','#d62728', '#9467bd']
for idx, (model_name, result) in enumerate(results.items()):
    plt.plot(mean_fpr, result["mean_tpr"],
             color=colors[idx],
             label=f'{model_name}',
             lw=2, alpha=.8)

    # 填充标准差
    std_tpr = np.std(result["interp_tprs"], axis=0)
    tprs_upper = np.minimum(result["mean_tpr"] + std_tpr, 1)
    tprs_lower = np.maximum(result["mean_tpr"] - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[idx], alpha=.15)

# 绘制对角线
plt.plot([0, 1], [0, 1], linestyle='--', color='#808080', lw=2, alpha=.8)

# 设置图形
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('FPR')
plt.ylabel('TPR')
#plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.savefig('SoA_ROC中文版.png')
plt.show()
