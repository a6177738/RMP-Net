import math

def calculate_mean(data):
    """
    计算数据的均值
    """
    return sum(data) / len(data)

def calculate_std_deviation(data):
    """
    计算数据的标准偏差
    """
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

#示例数据
with open('最高结果记录/mce.txt', 'r', encoding='utf-8') as file:
    # 读取前十行
    lines = [file.readline().strip() for _ in range(5)]


# 将每行按空格分割成元素
split_lines = [line.split() for line in lines]
Gmean = list()
F1 = list()
Sen = list()
Spe = list()
Mcc = list()
AUC = list()
for i in range(5):
    Gmean.append(float(split_lines[i][1]))
    F1.append(float(split_lines[i][7]))
    Sen.append(float(split_lines[i][9]))
    Spe.append(float(split_lines[i][11]))
    Mcc.append(float(split_lines[i][13]))
    AUC.append(float(split_lines[i][15]))

#计算均值
Gmean_mean = calculate_mean(Gmean)
F1_mean = calculate_mean(F1)
Sen_mean = calculate_mean(Sen)
Spe_mean = calculate_mean(Spe)
Mcc_mean = calculate_mean(Mcc)
AUC_mean = calculate_mean(AUC)
print('均值', Gmean_mean,  Sen_mean, Spe_mean, Mcc_mean,F1_mean, AUC_mean)

# 计算标准偏差
Gmean_std_deviation = calculate_std_deviation(Gmean)
F1_std_deviation = calculate_std_deviation(F1)
Sen_std_deviation = calculate_std_deviation(Sen)
Spe_std_deviation = calculate_std_deviation(Spe)
Mcc_std_deviation = calculate_std_deviation(Mcc)
AUC_std_deviation = calculate_std_deviation(AUC)

print("标准偏差", Gmean_std_deviation,Sen_std_deviation,Spe_std_deviation,Mcc_std_deviation,F1_std_deviation,AUC_std_deviation)