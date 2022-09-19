# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import Normalizer

def circRNASS(md_adjmat, disease_ss, *args, **kwargs):

    rows = np.size(md_adjmat, 0)    # should be 585
    result = np.zeros((rows, rows))

    for i in range(rows):
        idx = [idx for (idx, val) in enumerate(md_adjmat[i,:]) if val ==1]      #   找到关联矩阵中的 1
        if idx==False:
            continue
        for j in range(i):
            idy = [idy for (idy, val) in enumerate(md_adjmat[j, :]) if val == 1]    #    返回第j行的列索引
            if idy==False:
                continue

            sum1 = 0
            sum2 = 0

            for k1 in range(len(idx)):
                temp_max=0
                for d1 in range(len(idy)):
                    if disease_ss[idx[k1], idy[d1]]>temp_max:
                        temp_max=disease_ss[idx[k1], idy[d1]]
                sum1 = sum1 + temp_max  #   max(disease_ss(idx[k1], idy))


            for k2 in range(len(idy)):
                temp_max = 0
                for d2 in range(len(idx)):
                    if disease_ss[idy[k2], idx[d2]] > temp_max:
                        temp_max = disease_ss[idy[k2], idx[d2]]
                sum2 = sum2 + temp_max
            result[i, j] = (sum1 + sum2) / len(idx) + len(idy)
            result[j, i] = result[i, j]
        for k in range(rows):
            result[k, k] = 1
    return result

def normalize(arr,maxx,minn):
    arr1=np.copy(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr1[i][j] = (arr[i][j] -minn) / (maxx -minn)
        arr1[i][i]=1
    return arr1

def maxmin(A):
    maxx=-1000
    minn=1000
    for i in range (A.shape[0]):
        if max(A[i])>maxx:
            maxx=max(A[i])
        if min(A[i]) <minn:
            minn = min(A[i])
    return maxx, minn

if __name__ == '__main__':
    print("基于dis综合（GIP and disSemSim）相似度计算circRNA语义相似度")
    interactions_ori = pd.read_csv('../dataset/Association Matrix.txt', sep='\t', header=None)
    disSim_ori = pd.read_csv('../Feature/disSim_all.csv', encoding='gb18030', header=None).values

    circRNASS = circRNASS(np.array(interactions_ori),np.array(disSim_ori))
    # result = pd.DataFrame(np.array(circRNASS))
    # result.to_csv('../Feature/circFunSim.csv',header=False,index = False)

    maxx,minn=maxmin(np.array(circRNASS))
    circRNASS=normalize(np.array(circRNASS),maxx,minn)
    result = pd.DataFrame(circRNASS)
    result.to_csv('../Feature/circFunSim_norm.csv',header=False,index = False)





