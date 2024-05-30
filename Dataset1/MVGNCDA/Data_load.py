# Title     : Data_load
# Objective : TODO 正样本分为5折，取4折作为训练集，取1折作为测试集
# Objective : TODO 多加与4折相同数量的负样本和4折正样本组合作为测试集。
# Objective : TODO 将1折正样本和所有负样本作为测试集（包括之前提出的和4折相同数量的负样本）
# Created by: sunguicong
# Created on: 2021/10/21
import random

import numpy as np
import pandas as pd
import csv


path="../dataset/c_d.csv"



Rowid=[]
Cloumnid=[]
Labels=[]
Divide=[]
Rowid_neg=[]
Cloumnid_neg=[]
Labels_neg=[]
Divide_neg=[]

# 读取csv并保存正负样本
with open(path, 'r', newline='') as csv_file:
    reader = csv.reader(csv_file)
    tt=0
    for line in reader:
        for i in range(len(line)):
            if float(line[i])==1:
                Divide.append("222")        #正样本
                Rowid.append(tt)
                Cloumnid.append(i)
                Labels.append(int(float(line[i])))
            else:
                Divide_neg.append("111")        #负样本
                Rowid_neg.append(tt)
                Cloumnid_neg.append(i)
                Labels_neg.append(int(float(line[i])))
        tt=tt+1

# print(len(Rowid), len(Cloumnid), len(Labels), len(Divide))
# print(Rowid[0], Cloumnid[0], Labels[0], Divide[0])

#   整合正负样本的4列，并进行打乱
Data=[Rowid,Cloumnid,Labels,Divide]
Data_neg=[Rowid_neg,Cloumnid_neg,Labels_neg,Divide_neg]

Data=np.array(Data).T
Data_neg=np.array(Data_neg).T
print(Data.shape,Data_neg.shape)

row=list(range(Data.shape[0]))
random.shuffle(row)
Data=Data[row]


# Ten fold cross-validation
num_cross_val=5
for fold in range(num_cross_val):
    train_pos = np.array([x for i, x in enumerate(Data) if i % num_cross_val != fold])
    test_pos = np.array([x for i, x in enumerate(Data) if i % num_cross_val == fold])

    clo = list(range(Data_neg.shape[0]))
    random.shuffle(clo)
    num = Data.shape[0] / 5*4       # 取和4折正样本相同数量的负样本
    train_neg = Data_neg[clo][:int(num),:]

    #   重新设置训练和测试标签
    for i in range(train_neg.shape[0]):
        train_neg[i][3]="222"
    for i in range(test_pos.shape[0]):
        test_pos[i][3]="111"

    #   最终   组合训练和测试集
    train=np.concatenate((train_pos, train_neg), axis=0)
    test=np.concatenate((test_pos, Data_neg), axis=0)

    #再次打乱数据
    li = list(range(train.shape[0]))
    random.shuffle(li)
    train = train[li]
    li = list(range(test.shape[0]))
    random.shuffle(li)
    test = test[li]

    df = np.concatenate((train, test), axis=0)
    print(df.shape)

    df = pd.DataFrame({'Rowid':df[:,0],'Cloumnid':df[:,1],'Labels':df[:,2],'Divide':df[:,3]})
    file="../Feature/ff/cvfold"+str(fold)+".csv"
    df.to_csv(file,index=False,sep=',',header=False)


















