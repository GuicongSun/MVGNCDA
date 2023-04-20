# Title     : Save_data
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/10/22
import numpy as np
import pandas as pd
import scipy.sparse as sp
# In[]


#   全局归一化二部邻接矩阵集
def globally_normalize_bipartite_adjacency(adjacencies, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    print('{} normalizing bipartite adj'.format(['Asymmetrically', 'Symmetrically'][symmetric]))

    adj_tot = np.sum([adj for adj in adjacencies])
    # print(adj_tot)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    #   np.inf 置为无穷大
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(
            degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm

def get_adjacency(edge_df, num_lnc, num_dis, symmetric_normalization):
    lnc2dis_adjacencies = []
    dis2lnc_adjacencies = []
    train_edge_df = edge_df.loc[edge_df['usage'] == 'train']

    # print('-' * 200)
    # print(np.array(train_edge_df).shape)

    for i in range(2):
        edge_index = train_edge_df.loc[train_edge_df.label == i, ['circid', 'disid']].to_numpy()

        # print(edge_index.shape)
        support = sp.csr_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),shape=(num_lnc, num_dis), dtype=np.float32)
            # print(support.shape)

        lnc2dis_adjacencies.append(support)
        dis2lnc_adjacencies.append(support.T)


    #   全局归一化二部邻接矩阵集
    lnc2dis_adjacencies = globally_normalize_bipartite_adjacency(lnc2dis_adjacencies, symmetric=symmetric_normalization)

    dis2lnc_adjacencies = globally_normalize_bipartite_adjacency(dis2lnc_adjacencies,
                                                                 symmetric=symmetric_normalization)

    return lnc2dis_adjacencies, dis2lnc_adjacencies

def get_node_identity_feature(num_lnc, num_dis):
    """one-hot encoding for nodes"""

    identity_feature = np.identity(num_lnc + num_dis, dtype=np.float32)
    lnc_identity_feature, dis_indentity_feature = identity_feature[:num_lnc], identity_feature[num_lnc:]

    return lnc_identity_feature, dis_indentity_feature


def build_graph_adj(edge_df, symmetric_normalization=False):
    #   计算出lncrna和disease的数量
    node_lnc = edge_df[['circid']].drop_duplicates().sort_values('circid')  # (240,1)
    node_movie = edge_df[['disid']].drop_duplicates().sort_values('disid')  # (412,1)

    num_lnc = len(node_lnc)
    num_dis = len(node_movie)
    print("circ number：",num_lnc,"dis number：",num_dis)

    # adjacency     #   得到归一化后的二部邻接矩阵集          以压缩稀疏行格式存储 520 个元素  *2
    lnc2dis_adjacencies, dis2lnc_adjacencies = get_adjacency(edge_df, num_lnc, num_dis, symmetric_normalization)

    print('-'*100)
    # one-hot encoding for nodes        分别为       (585, 673)                        (88, 673)
    lnc_identity_feature, dis_indentity_feature = get_node_identity_feature(num_lnc, num_dis)

    print(lnc_identity_feature.shape,dis_indentity_feature.shape)
    #   将会用到的train样本和test样本标记为True ，否则为False
    # lnc_indices, dis_indices, labels, train_mask
    lnc_indices, dis_indices, labels = edge_df[['circid', 'disid', 'label']].to_numpy().T
    train_mask = (edge_df['usage'] == 'train').to_numpy()
    test_mask = (edge_df['usage'] == 'test').to_numpy()

    return lnc2dis_adjacencies, dis2lnc_adjacencies, \
           lnc_identity_feature, dis_indentity_feature, \
           lnc_indices, dis_indices, labels, train_mask, test_mask


def read_edge2(filename):
    edge_df = pd.read_csv('./' + filename, header=None)
    # print(edge_df.shape)

    columns2 = ['circid', 'disid','label','usage']
    for i in range(4):
        edge_df.rename(columns={edge_df.columns[i]: columns2[i]}, inplace=True)
    # print(edge_df)

    edge_df.loc[edge_df['usage'] == 222, 'usage'] = 'train'
    edge_df.loc[edge_df['usage'] == 111, 'usage'] = 'test'


    return edge_df



U2madj = []
M2uadj = []
Uidf = []
Midf = []
Uind = []
Mind = []
Label = []
Tramask = []
Testmask = []
#
num_cross_val=5
for i in range(num_cross_val):
    filename = '../Feature/ff/cvfold' + str(i) + '.csv'
    edge_df = read_edge2(filename=filename)
    lnc2dis_adjacencies, dis2lnc_adjacencies, \
    lnc_identity_feature, dis_indentity_feature, \
    lnc_indices, dis_indices, labels, train_mask, test_mask = build_graph_adj(edge_df=edge_df,symmetric_normalization=False)


    U2madj.append(lnc2dis_adjacencies)
    M2uadj.append(dis2lnc_adjacencies)
    Uidf.append(lnc_identity_feature)
    Midf.append(dis_indentity_feature)
    Uind.append(lnc_indices)
    Mind.append(dis_indices)
    Label.append(labels)
    Tramask.append(train_mask)
    Testmask.append(test_mask)
#


# 保存数据
import pickle

output = open('../Feature/5fold_data.pkl', 'wb')
pickle.dump(U2madj, output)
pickle.dump(M2uadj, output)
pickle.dump(Uidf, output)
pickle.dump(Midf, output)
pickle.dump(Uind, output)
pickle.dump(Mind, output)
pickle.dump(Label, output)
pickle.dump(Tramask, output)
pickle.dump(Testmask, output)
output.close()
