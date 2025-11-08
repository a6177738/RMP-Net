import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成示例数据
mu_modality1 = 5
theta_modality1 = 1

mu_modality2 = 6
theta_modality2 = 1.2

mu_fusion = (mu_modality1 + mu_modality2) / 2
theta_fusion = (theta_modality1 + theta_modality2) / 2

# 定义x轴范围
x = np.linspace(0, 10, 1000)

# 计算概率分布
pdf_modality1 = norm.pdf(x, mu_modality1, theta_modality1)
pdf_modality2 = norm.pdf(x, mu_modality2, theta_modality2)
pdf_fusion = norm.pdf(x, mu_fusion, theta_fusion)

# 使用 seaborn 绘制染色的概率分布图
plt.figure(figsize=(12, 8))

sns.lineplot(x=x, y=pdf_modality1, label='Modality 1', color='orange')
sns.lineplot(x=x, y=pdf_modality2, label='Modality 2', color='blue')
sns.lineplot(x=x, y=pdf_fusion, label='Fusion', color='green')

plt.fill_between(x, pdf_modality1, alpha=0.3, color='orange')
plt.fill_between(x, pdf_modality2, alpha=0.3, color='blue')
plt.fill_between(x, pdf_fusion, alpha=0.3, color='green')

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Probability Distributions for a Single Sample')
plt.legend()

# 移除背景网格
sns.despine()
plt.grid(False)
plt.savefig('1.png')

plt.show()
