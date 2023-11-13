# Title     : Model
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/11/1

import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
torch.backends.cudnn.enabled = False
import csv
from sklearn.decomposition import PCA




class Decoder(nn.Module):
    def __init__(self, input_dim, num_weights, num_classes, dropout=0., activation=F.relu):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.num_weights = num_weights
        self.num_classes = num_classes
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(num_weights, input_dim, input_dim))
        self.weight_classifier = nn.Parameter(torch.Tensor(num_weights, num_classes))
        self.reset_parameters()

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        init.kaiming_uniform_(self.weight_classifier)
        init.kaiming_uniform_(self.weight_classifier)

    def forward(self, circ_inputs, dis_inputs, circ_indices, dis_indices):

        circ_inputs = circ_inputs[circ_indices]
        dis_inputs = dis_inputs[dis_indices]

        basis_outputs = []
        for i in range(self.num_weights):
            tmp = torch.matmul(circ_inputs, self.weight[i])
            out = torch.sum(tmp * dis_inputs, dim=1, keepdim=True)
            basis_outputs.append(out)

        basis_outputs = torch.cat(basis_outputs, dim=1)
        outputs = torch.matmul(basis_outputs, self.weight_classifier)
        outputs = self.activation(outputs)
        return outputs


class CDA(nn.Module):
    def __init__(self,gcn_layers = 3,out_channels = 64,circRNA_number = 585,fc = 128,disease_number = 88, fd = 128, view = 2 ):
        super(CDA, self).__init__()

        self.gcn_layers = gcn_layers
        self.out_channels = out_channels
        self.circRNA_number = circRNA_number
        self.fc = fc
        self.disease_number = disease_number
        self.fd = fd
        self.view = view


        # circRNA
        self.gcn_x1_g=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_x1_c=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_x1_s=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_circSim1=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_x2_c=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_x2_s=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_x2_g=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_circSim2=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_x3_c=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_x3_s=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_x3_g=GCNConv(self.fc, self.fc,cached=True)
        self.gcn_circSim3=GCNConv(self.fc, self.fc,cached=True)
        # disease
        self.gcn_y1_g=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_y1_c=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_y1_s=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_disSim1=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_y2_g=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_y2_c=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_y2_s=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_disSim2=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_y3_g=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_y3_c=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_y3_s=GCNConv(self.fd, self.fd,cached=True)
        self.gcn_disSim3=GCNConv(self.fd, self.fd,cached=True)

        self.decoder = Decoder(128, 2, 2)

        self.pca1=PCA(n_components=64)
        self.pca2=PCA(n_components=64)



    def forward (self,data,Gra_emb_dis,Gra_emb_circrna,circ_index,dis_index):
        # Graph embedding
        Gra_emb_dis = torch.Tensor.cpu(torch.from_numpy(np.array(Gra_emb_dis)).float()).cuda()
        Gra_emb_circrna = torch.Tensor.cpu(torch.from_numpy(np.array(Gra_emb_circrna)).float()).cuda()


        torch.manual_seed(-1000)
        x_fc = torch.randn(self.circRNA_number, self.fc) # torch.Size([585, 256])
        y_fd = torch.randn(self.disease_number, self.fd) # torch.Size([88, 256])

###     分别基于GIP；cosine；语义相似度；集成相似度
#       传入矩阵，索引（非0），值（非0）
        x_c_g1 = torch.relu(self.gcn_x1_g(x_fc.cuda(), data['cc_g']['edges'].cuda(), data['cc_g']['data_matrix'][data['cc_g']['edges'][0], data['cc_g']['edges'][1]].cuda()))
        x_c_g2 = torch.relu(self.gcn_x2_g(x_c_g1, data['cc_g']['edges'].cuda(), data['cc_g']['data_matrix'][data['cc_g']['edges'][0], data['cc_g']['edges'][1]].cuda()))
        x_c_g3 = torch.relu(self.gcn_x3_g(x_c_g2, data['cc_g']['edges'].cuda(), data['cc_g']['data_matrix'][data['cc_g']['edges'][0], data['cc_g']['edges'][1]].cuda()))

        x_c_c1 = torch.relu(self.gcn_x1_c(x_fc.cuda(), data['cc_c']['edges'].cuda(), data['cc_c']['data_matrix'][data['cc_c']['edges'][0], data['cc_c']['edges'][1]].cuda()))
        x_c_c2 = torch.relu(self.gcn_x2_c(x_c_c1, data['cc_c']['edges'].cuda(), data['cc_c']['data_matrix'][data['cc_c']['edges'][0], data['cc_c']['edges'][1]].cuda()))
        x_c_c3 = torch.relu(self.gcn_x3_c(x_c_c2, data['cc_c']['edges'].cuda(), data['cc_c']['data_matrix'][data['cc_c']['edges'][0], data['cc_c']['edges'][1]].cuda()))

        x_c_s1 = torch.relu(self.gcn_x1_s(x_fc.cuda(), data['cc_sem']['edges'].cuda(), data['cc_sem']['data_matrix'][data['cc_sem']['edges'][0], data['cc_sem']['edges'][1]].cuda()))
        x_c_s2 = torch.relu(self.gcn_x2_s(x_c_s1, data['cc_sem']['edges'].cuda(), data['cc_sem']['data_matrix'][data['cc_sem']['edges'][0], data['cc_sem']['edges'][1]].cuda()))
        x_c_s3 = torch.relu(self.gcn_x3_s(x_c_s2, data['cc_sem']['edges'].cuda(), data['cc_sem']['data_matrix'][data['cc_sem']['edges'][0], data['cc_sem']['edges'][1]].cuda()))

        circSim1 = torch.relu(self.gcn_circSim1(x_fc.cuda(), data['circSim']['edges'].cuda(), data['circSim']['data_matrix'][data['circSim']['edges'][0], data['circSim']['edges'][1]].cuda()))
        circSim2 = torch.relu(self.gcn_circSim2(circSim1, data['circSim']['edges'].cuda(), data['circSim']['data_matrix'][data['circSim']['edges'][0], data['circSim']['edges'][1]].cuda()))
        circSim3 = torch.relu(self.gcn_circSim3(circSim2, data['circSim']['edges'].cuda(), data['circSim']['data_matrix'][data['circSim']['edges'][0], data['circSim']['edges'][1]].cuda()))


        y_d_g1 = torch.relu(self.gcn_y1_g(y_fd.cuda(), data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
        y_d_g2 = torch.relu(self.gcn_y2_g(y_d_g1, data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
        y_d_g3 = torch.relu(self.gcn_y3_g(y_d_g2, data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))

        y_d_c1 = torch.relu(self.gcn_y1_c(y_fd.cuda(), data['dd_c']['edges'].cuda(), data['dd_c']['data_matrix'][data['dd_c']['edges'][0], data['dd_c']['edges'][1]].cuda()))
        y_d_c2 = torch.relu(self.gcn_y2_c(y_d_c1, data['dd_c']['edges'].cuda(), data['dd_c']['data_matrix'][data['dd_c']['edges'][0], data['dd_c']['edges'][1]].cuda()))
        y_d_c3 = torch.relu(self.gcn_y3_c(y_d_c2, data['dd_c']['edges'].cuda(), data['dd_c']['data_matrix'][data['dd_c']['edges'][0], data['dd_c']['edges'][1]].cuda()))

        y_d_s1 = torch.relu(self.gcn_y1_s(y_fd.cuda(), data['dd_dag']['edges'].cuda(), data['dd_dag']['data_matrix'][data['dd_dag']['edges'][0], data['dd_dag']['edges'][1]].cuda()))
        y_d_s2 = torch.relu(self.gcn_y2_s(y_d_s1, data['dd_dag']['edges'].cuda(), data['dd_dag']['data_matrix'][data['dd_dag']['edges'][0], data['dd_dag']['edges'][1]].cuda()))
        y_d_s3 = torch.relu(self.gcn_y3_s(y_d_s2, data['dd_dag']['edges'].cuda(), data['dd_dag']['data_matrix'][data['dd_dag']['edges'][0], data['dd_dag']['edges'][1]].cuda()))

        disSim1 = torch.relu(self.gcn_disSim1(y_fd.cuda(), data['disSim']['edges'].cuda(), data['disSim']['data_matrix'][data['disSim']['edges'][0], data['disSim']['edges'][1]].cuda()))
        disSim2 = torch.relu(self.gcn_disSim2(disSim1, data['disSim']['edges'].cuda(), data['disSim']['data_matrix'][data['disSim']['edges'][0], data['disSim']['edges'][1]].cuda()))
        disSim3 = torch.relu(self.gcn_disSim3(disSim2, data['disSim']['edges'].cuda(), data['disSim']['data_matrix'][data['disSim']['edges'][0], data['disSim']['edges'][1]].cuda()))




        if (self.gcn_layers==1):
            XM = torch.cat((x_c_g1, x_c_c1, x_c_s1), 1)
            YD = torch.cat((y_d_g1, y_d_c1, y_d_s1), 1)
        elif(self.gcn_layers==2):
            XM = torch.cat((x_c_g1, x_c_g2, x_c_c1, x_c_c2,x_c_s1,x_c_s2), 1)
            YD = torch.cat((y_d_g1, y_d_g2, y_d_c1, y_d_c2,y_d_s1,y_d_s2), 1)
        else:
            XM = torch.cat((x_c_g1, x_c_g2, x_c_g3, x_c_c1, x_c_c2, x_c_c3,x_c_s1,x_c_s2,x_c_s3), 1)
            YD = torch.cat((y_d_g1, y_d_g2, y_d_g3, y_d_c1, y_d_c2, y_d_c3,y_d_s1,y_d_s2,y_d_s3), 1)

        XM = self.pca1.fit_transform(XM.cpu().detach().numpy())
        YD = self.pca2.fit_transform(YD.cpu().detach().numpy())

        XM=torch.Tensor.cpu(torch.from_numpy(XM)).cuda()
        YD=torch.Tensor.cpu(torch.from_numpy(YD)).cuda()

        # p=2表示二范式, dim=1表示按行归一化
        XM = F.normalize(XM, p=2, dim=1)
        YD = F.normalize(YD, p=2, dim=1)



        ### 两者结合使用
        XM = torch.cat((XM, Gra_emb_circrna), 1)
        YD = torch.cat((YD, Gra_emb_dis), 1)

        outputs = self.decoder(XM, YD,circ_index,dis_index)
        return outputs
        # return x.mm(y.t())









