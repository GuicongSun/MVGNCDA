# Title     : txt_2_csv
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/10/17
import numpy as np
import pandas as pd
import math




CDA_matrix = pd.read_csv('c_d.txt', sep='\t', header=None)

CDA_matrix=np.array(CDA_matrix)
# circRNA和diease的数量
nc=np.array(CDA_matrix).shape[0]
nd=np.array(CDA_matrix).shape[1]

circrna_similarity= np.zeros([nc, nd])  # 行与行之间的相似度，初始化矩阵


for i in range(nc):
    for j in range(nd):
        circrna_similarity[i,j]=int(CDA_matrix[i,j])

# 保存结果
result = pd.DataFrame(circrna_similarity)
result.to_csv('c_d.csv',header=False,index = False)