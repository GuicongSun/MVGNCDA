import csv
import torch
import random
import numpy as np

def metric_calculate_performace(test_num, pred_y, labels):  # pred_y = proba, labels = real_labels
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num

    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score = 0
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        f1_score = float(2 * tp) / ((2 * tp) + fp + fn)


    return acc, precision, sensitivity, specificity, MCC, f1_score,tp,fn,tn,fp


#   将关联矩阵中的每个数据转化为tensor后读出
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


#   获取有联系的边的索引       一维是横坐标，二维是纵坐标
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


#读取辅助（相似性）信息
def data_pro(dataset_path):
    dataset = dict()

    #   导出节点和边索引
    "disease gip sim"
    dd_g_matrix = read_csv(dataset_path + '/disGIPSim.csv')
    dd_g_edge_index = get_edge_index(dd_g_matrix)
    dataset['dd_g'] = {'data_matrix': dd_g_matrix, 'edges': dd_g_edge_index}

    "disease consine sim"
    dd_c_matrix = read_csv(dataset_path + '/disCosSim.csv')
    # dd_c_matrix = read_csv(dataset_path + '/ddd.txt')
    dd_c_edge_index = get_edge_index(dd_c_matrix)
    dataset['dd_c'] = {'data_matrix': dd_c_matrix, 'edges': dd_c_edge_index}

    "disease DAG sim"
    dd_dag_matrix = read_csv(dataset_path + '/disSSim.csv')
    dd_dag_edge_index = get_edge_index(dd_dag_matrix)
    dataset['dd_dag'] = {'data_matrix': dd_dag_matrix, 'edges': dd_dag_edge_index}

    "circRNA gip sim"
    cc_g_matrix = read_csv(dataset_path + '/circGIPSim.csv')
    cc_g_edge_index = get_edge_index(cc_g_matrix)
    dataset['cc_g'] = {'data_matrix': cc_g_matrix, 'edges': cc_g_edge_index}

    "circRNA consine sim"
    cc_c_matrix = read_csv(dataset_path + '/circCosSim.csv')
    # cc_c_matrix = read_csv(dataset_path + '/ccc.txt')
    cc_c_edge_index = get_edge_index(cc_c_matrix)
    dataset['cc_c'] = {'data_matrix': cc_c_matrix, 'edges': cc_c_edge_index}

    "circRNA sem sim"
    cc_sem_matrix = read_csv(dataset_path + '/circFunSim_norm.csv')   #   circRNASS
    cc_sem_edge_index = get_edge_index(cc_sem_matrix)
    dataset['cc_sem'] = {'data_matrix': cc_sem_matrix, 'edges': cc_sem_edge_index}

    # intergate embedding
    "disease sim"
    disSim = read_csv(dataset_path + '/disSim_all.csv')
    cc_g_edge_index = get_edge_index(disSim)
    dataset['disSim'] = {'data_matrix': disSim, 'edges': cc_g_edge_index}

    "circRNA  sim"
    circSim = read_csv(dataset_path + '/circSim_all.csv')
    cc_c_edge_index = get_edge_index(circSim)
    dataset['circSim'] = {'data_matrix': circSim, 'edges': cc_c_edge_index}

    return dataset

