'''

使用疾病语义相似度补全GIP相似度矩阵。
即将GIP相似度矩阵中的部分值 替换为 疾病语义相似度特征（如果有）。

'''


import numpy as np
import pandas as pd

# 读取数据

similarity1 = pd.read_csv('../Feature/disSSim_1.csv', header=None, encoding='gb18030').values
similarity2 = pd.read_csv('../Feature/disSSim_2.csv', header=None, encoding='gb18030').values

similarity1 = np.mat(similarity1)
similarity2 = np.mat(similarity2)


print(similarity1.shape)
# 矩阵尺寸为88行88列

disSSim=similarity1

for m in range(disSSim.shape[0]):
    for n in range(disSSim.shape[1]):
        # if(similarity1[m, n]==0 and similarity2[m, n]!=0):
        #     print("这不可能")
        disSSim[m, n] = (similarity1[m, n] + similarity2[m, n])*0.5
        if m == n:
            disSSim[m, n] = 0.8


# 保存结果

result = pd.DataFrame(disSSim)
result.to_csv('../Feature/disSSim.csv',header=False,index = False)



