#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Title     : node2vec_preprocess.py
# Objective : TODO  将关联矩阵转化为图嵌入所需要的边列表，后续还需手动操作
# Created by: sunguicong
# Created on: 2022/5/11

import numpy as np
import pandas as pd
import math
import csv



CDA_adj = pd.read_csv('../dataset/Association Matrix.txt', sep='\t', header=None)

CDA_adj=np.array(CDA_adj)
# circRNA和diease的数量
nc=np.array(CDA_adj).shape[0]   #585
nd=np.array(CDA_adj).shape[1]   #88

circrna_similarity= np.zeros([nc, nd])  # 行与行之间的相似度，初始化矩阵

with open("../Feature/adj_eagelist.txt", "w") as csvfile:

    for i in range(nc):
        for j in range(nd):
            if CDA_adj[i][j]==1:
                # csvfile.writerow([i+nd, j])
                csvfile.writelines([str(i+nd), " ", str(j), '\n'])


