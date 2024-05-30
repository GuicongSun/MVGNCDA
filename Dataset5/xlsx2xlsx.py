# -*- coding: utf-8 -*-

import pandas as pd

# 加载Excel文件
file_path = '1.The circRNA-disease entries.xlsx'
xl = pd.ExcelFile(file_path)

# 读取三个附表
df_associations = xl.parse('The CircRNA-Disease Entries',header=0)  # 假设关联在附表1
df_diseases = xl.parse('disease list',header=None)      # 假设疾病在附表2
df_circRNAs = xl.parse('circRNA list',header=None)      # 假设circRNA在附表3

# 获取疾病和circRNA的列表
diseases = df_diseases.iloc[:, 0].str.lower().tolist()  # 假设疾病名在附表2的第一列
circRNAs = df_circRNAs.iloc[:, 0].str.lower().tolist()  # 假设circRNA名在附表3的第一列

# 创建邻接矩阵的零矩阵
adj_matrix = pd.DataFrame(0, index=circRNAs, columns=diseases)

i=0
# 填充邻接矩阵
for _, row in df_associations.iterrows():
    i = i + 1
    circRNA = row[1].lower()  # 假设circRNA名称在附表1的第2列
    disease = row[4].lower()  # 假设疾病名称在附表1的第5列
    if circRNA in circRNAs and disease in diseases:
        if adj_matrix.at[circRNA, disease] == 1:
            print(i)
            # print(circRNA, disease)
            # print("GG")
        adj_matrix.at[circRNA, disease] = 1
    else:
        print('-'*10)
        print(circRNA)
        print(disease)
        print('='*10)


# 显示邻接矩阵
print(adj_matrix)

# 如果需要保存这个矩阵到新的Excel文件
adj_matrix.to_excel('adjacency_matrix.xlsx')



