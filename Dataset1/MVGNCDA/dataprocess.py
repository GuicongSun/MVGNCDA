# Title     : dataprocess
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/10/23
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from numpy import interp

from sklearn.metrics import roc_curve, auc
import scipy.sparse as sp

def to_torch_sparse_tensor(x):
    if not sp.isspmatrix_coo(x):
        x = sp.coo_matrix(x)
    row, col = x.row, x.col
    indices = torch.from_numpy(np.asarray([row, col]).astype('int64')).long()
    values = torch.from_numpy(x.data.astype(np.float32))
    th_sparse_tensor = torch.sparse.FloatTensor(indices, values,
                                                x.shape)
    return th_sparse_tensor

def tensor_from_numpy(x):
    return torch.from_numpy(x)



def data_5fold(fold):

    import pickle
    pkl_file = open('../Feature/5fold_data.pkl', 'rb')
    U2madj = pickle.load(pkl_file)
    M2uadj = pickle.load(pkl_file)
    Uidf = pickle.load(pkl_file)
    Midf = pickle.load(pkl_file)
    Uind = pickle.load(pkl_file)
    Mind = pickle.load(pkl_file)
    Label = pickle.load(pkl_file)
    Tramask = pickle.load(pkl_file)
    Testmask = pickle.load(pkl_file)
    pkl_file.close()


    lnc_indices = Uind[0]
    lnc_indices2 = Uind[1]
    lnc_indices3 = Uind[2]
    lnc_indices4 = Uind[3]
    lnc_indices5 = Uind[4]

    dis_indices = Mind[0]
    dis_indices2 = Mind[1]
    dis_indices3 = Mind[2]
    dis_indices4 = Mind[3]
    dis_indices5 = Mind[4]

    labels = Label[0]
    labels2 = Label[1]
    labels3 = Label[2]
    labels4 = Label[3]
    labels5 = Label[4]

    train_mask = Tramask[0]
    train_mask2 = Tramask[1]
    train_mask3 = Tramask[2]
    train_mask4 = Tramask[3]
    train_mask5 = Tramask[4]

    test_mask = Testmask[0]
    test_mask2 = Testmask[1]
    test_mask3 = Testmask[2]
    test_mask4 = Testmask[3]
    test_mask5 = Testmask[4]

    lnc_indices1 = tensor_from_numpy(lnc_indices).long()
    dis_indices1 = tensor_from_numpy(dis_indices).long()
    labels1 = tensor_from_numpy(labels)
    train_mask1 = tensor_from_numpy(train_mask)
    test_mask1 = tensor_from_numpy(test_mask)

    lnc_indices2 = tensor_from_numpy(lnc_indices2).long()
    dis_indices2 = tensor_from_numpy(dis_indices2).long()
    labels2 = tensor_from_numpy(labels2)
    train_mask2 = tensor_from_numpy(train_mask2)
    test_mask2 = tensor_from_numpy(test_mask2)

    lnc_indices3 = tensor_from_numpy(lnc_indices3).long()
    dis_indices3 = tensor_from_numpy(dis_indices3).long()
    labels3 = tensor_from_numpy(labels3)
    train_mask3 = tensor_from_numpy(train_mask3)
    test_mask3 = tensor_from_numpy(test_mask3)

    lnc_indices4 = tensor_from_numpy(lnc_indices4).long()
    dis_indices4 = tensor_from_numpy(dis_indices4).long()
    labels4 = tensor_from_numpy(labels4)
    train_mask4 = tensor_from_numpy(train_mask4)
    test_mask4 = tensor_from_numpy(test_mask4)

    lnc_indices5 = tensor_from_numpy(lnc_indices5).long()
    dis_indices5 = tensor_from_numpy(dis_indices5).long()
    labels5 = tensor_from_numpy(labels5)
    train_mask5 = tensor_from_numpy(train_mask5)
    test_mask5 = tensor_from_numpy(test_mask5)

    if fold == 0:
        return lnc_indices1,dis_indices1,train_mask1,test_mask1,labels1
    elif fold == 1:
        return lnc_indices2,dis_indices2,train_mask2,test_mask2,labels2
    elif fold == 2:
        return lnc_indices3,dis_indices3,train_mask3,test_mask3,labels3
    elif fold == 3:
        return lnc_indices4,dis_indices4,train_mask4,test_mask4,labels4
    elif fold == 4:
        return lnc_indices5,dis_indices5,train_mask5,test_mask5,labels5




