import numpy as np
import pandas as pd


# 数据格式如下
# disease = ["Meningitis, Pneumococcal", "Meningitis, Pneumococcal", "Mycetoma", "Botulism", "Botulism2", "Botulism3"]
# id = ["C01.252.200.500.600", "C08.345.654.570", "C01.252.410.040.692.606", "C01.252.410.222.151", "C01.252.410.222.151", "C03.252.410.222.151"]

print("开始读取数据....")
# 读取数据
meshid = pd.read_csv('../dataset/MeSHID_2022.csv', header=0)
disease = meshid['disease'].tolist()
id = meshid['ID'].tolist()

meshdis = pd.read_csv('../dataset/Mesh_disease.csv', header=0)
unique_disease = meshdis['C1'].tolist()

# 初始化字典，有重复也没关系
for i in range(len(disease)):
    disease[i] = {}

print("开始计算每个病的DV")
# 计算每个病的DV，又重复也没关系，之后再合并

#   从ID的末尾开始，逐步往前递归计算贡献度，
for i in range(len(disease)):

    if len(id[i]) > 3:
        disease[i][id[i]] = 1
        id[i] = id[i][:-4]
        # print(disease[i])
        if len(id[i]) > 3:
            disease[i][id[i]] = round(1 * 0.8, 5)
            id[i] = id[i][:-4]
            # print(disease[i])
            if len(id[i]) > 3:
                disease[i][id[i]] = round(1 * 0.8 * 0.8, 5)
                id[i] = id[i][:-4]
                # print(disease[i])
                if len(id[i]) > 3:
                    disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8, 5)
                    id[i] = id[i][:-4]
                    # print(disease[i])
                    if len(id[i]) > 3:
                        disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                        id[i] = id[i][:-4]
                        # print(disease[i])
                        if len(id[i]) > 3:
                            disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                            id[i] = id[i][:-4]
                            # print(disease[i])
                            if len(id[i]) > 3:
                                disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                id[i] = id[i][:-4]
                                # print(disease[i])
                                if len(id[i]) > 3:
                                    disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                    id[i] = id[i][:-4]
                                    # print(disease[i])
                                else:
                                    disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                    # print(disease[i])
                            else:
                                disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                # print(disease[i])
                        else:
                            disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                            # print(disease[i])
                    else:
                        disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                        # print(disease[i])
                else:
                    disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8, 5)
                    # print(disease[i])
            else:
                disease[i][id[i][:3]] = round(1 * 0.8 * 0.8, 5)
                # print(disease[i])
        else:
            disease[i][id[i][:3]] = round(1 * 0.8, 5)
            # print(disease[i])
    else:
        disease[i][id[i][:3]] = 1
        # print(disease[i])

print("合并相同的病不同ID的DV")


#################################  疾病的语义值 semantic value of a disease ####################################
# 合并相同的病不同ID的DV

unique_disease = meshdis['C1'].tolist()

# 这个name用来判断
disease_name = meshid['disease'].tolist()
unique_disease_name = meshdis['C1'].tolist()

for i in range(len(unique_disease)):
    unique_disease[i] = {}
    for j in range(len(disease_name)):
        if unique_disease_name[i] == disease_name[j]:       # 如果当前疾病和MeSHID的疾病名称一样，则
            unique_disease[i].update(disease[j])        #   相当于把之前同一疾病所有的关联树都整合在了一起
    # print(unique_disease[i])



#################################  疾病的语义值 semantic value of a disease ####################################
similarity = np.zeros([len(unique_disease_name), len(unique_disease_name)])
# print(similarity)
print(similarity.shape)

print("计算相似度")
print(len(unique_disease_name))

for m in range(len(unique_disease_name)):
    for n in range(len(unique_disease_name)):
        denominator = sum(unique_disease[m].values()) + sum(unique_disease[n].values())     # 分母
        numerator = 0
        for k, v in unique_disease[m].items():  #   迭代疾病1的所有祖先节点    相同则相加
            if k in unique_disease[n].keys():   #   判断两种疾病的祖先节点是否一样，
                numerator += v + unique_disease[n].get(k)       # 分子    values相加
        if(denominator==0):
            if(m==n):
                similarity[m, n] =1
            else:
                similarity[m, n]=0
        else:
            similarity[m, n] = round(numerator/denominator, 5)

# print(similarity)
print("保存结果")

# 保存结果

result = pd.DataFrame(similarity)
result.to_csv('../Feature/disSSim_1.csv',header=False,index = False)
