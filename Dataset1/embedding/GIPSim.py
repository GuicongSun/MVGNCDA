# Title     : GIP
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/10/12

import numpy as np
import pandas as pd
import math


# CDA_matrix = pd.read_csv('Association Matrix.csv', header=None, encoding='gb18030').values

CDA_matrix = pd.read_csv('../dataset/Association Matrix.txt', sep='\t', header=None)


CDA_matrix=np.array(CDA_matrix)

print(np.array(CDA_matrix).shape)

# circRNA和diease的数量
nc=np.array(CDA_matrix).shape[0]
nd=np.array(CDA_matrix).shape[1]



## 计算circRNA之间的高斯相互作用相似度

circrna_similarity= np.zeros([nc, nc])  # 行与行之间的相似度，初始化矩阵

normSum = 0
for i in range(nc):
    normSum += np.sum(CDA_matrix[i]**2)**0.5**2  # 按定义用二阶范数计算
print(normSum)

for i in range(nc):
    for j in range(nc):
        circrna_similarity[i, j] = math.exp((np.sum((CDA_matrix[i] - CDA_matrix[j])**2)**0.5**2)  * normSum/nc* (-1))
        if circrna_similarity[i, j] == 1:
            circrna_similarity[i, j] = 0.8  # 这里是一个大问题，两个向量相同可以说它有一定相关度，可是计算出相关度等于1又不合理，只能定义一个值

# 保存结果
result = pd.DataFrame(circrna_similarity)
result.to_csv('../Feature/circGIPSim.csv',header=False,index = False)
# 注意，这样保存之后会多了一行一列行号序号，需要删除






## 计算disease之间的高斯相互作用相似度

CDA_matrix = CDA_matrix.T  # 转置方便计算

print(np.array(CDA_matrix).shape)

diease_similarity= np.zeros([nd, nd])  # 行与行之间的相似度，初始化矩阵

normSum = 0
for i in range(nd):
    normSum += np.sum(CDA_matrix[i]**2)**0.5**2  # 按定义用二阶范数计算
print(normSum)

for i in range(nd):
    for j in range(nd):
        diease_similarity[i, j] = math.exp((np.sum((CDA_matrix[i] - CDA_matrix[j])**2)**0.5**2)  * normSum/ nd * (-1))
        if diease_similarity[i, j] == 1:
            diease_similarity[i, j] = 0.8  # 这里是一个大问题，两个向量相同可以说它有一定相关度，可是计算出相关度等于1又不合理，只能定义一个值

# 保存结果
result = pd.DataFrame(diease_similarity)
result.to_csv('../Feature/disGIPSim.csv',header=False,index = False)
# 注意，这样保存之后会多了一行一列行号序号，需要删除







