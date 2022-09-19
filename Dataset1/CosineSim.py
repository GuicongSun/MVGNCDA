# Title     : Cosine
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/10/13



import numpy as np
import pandas as pd
import math


CDA_matrix = pd.read_csv('../dataset/Association Matrix.txt', sep='\t', header=None)

CDA_matrix=np.array(CDA_matrix)

print(np.array(CDA_matrix).shape)

# circRNA和diease的数量
nc=np.array(CDA_matrix).shape[0]
nd=np.array(CDA_matrix).shape[1]



## 计算circRNA之间的Cosine相似度

circrna_similarity= np.zeros([nc, nc])  # 行与行之间的相似度，初始化矩阵


for i in range(nc):
    for j in range(i):
        Denominator=(np.sum(CDA_matrix[i]**2)**0.5)*(np.sum(CDA_matrix[j]**2)**0.5) #计算cosine相似度公式的分母
        if (Denominator==0):
            circrna_similarity[i,j]=0
        else:
            circrna_similarity[i, j] = np.dot(CDA_matrix[i],CDA_matrix[j])/Denominator
        circrna_similarity[j,i]=circrna_similarity[i, j]
    circrna_similarity[i,i]=1

# 保存结果
result = pd.DataFrame(circrna_similarity)
result.to_csv('../Feature/circCosSim.csv',header=False,index = False)






## 计算disease之间的Cosine相似度

CDA_matrix = CDA_matrix.T  # 转置方便计算

print(np.array(CDA_matrix).shape)

disease_similarity= np.zeros([nd, nd])  # 行与行之间的相似度，初始化矩阵


for i in range(nd):
    for j in range(i):
        Denominator=(np.sum(CDA_matrix[i]**2)**0.5)*(np.sum(CDA_matrix[j]**2)**0.5) #计算cosine相似度公式的分母
        if (Denominator==0):
            disease_similarity[i,j]=0
        else:
            disease_similarity[i, j] = np.dot(CDA_matrix[i],CDA_matrix[j])/Denominator
        disease_similarity[j,i]=disease_similarity[i, j]
    disease_similarity[i,i]=1
# 保存结果
result = pd.DataFrame(disease_similarity)
result.to_csv('../Feature/disCosSim.csv',header=False,index = False)










