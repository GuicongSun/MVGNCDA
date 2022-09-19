# -*- coding: utf-8 -*-
import torch
# from param import parameter_parser
from util import data_pro
from util import read_csv
from util import metric_calculate_performace
import torch
import torch.nn.functional as F
from dataprocess import data_5fold
import numpy as np
from sklearn import metrics
from numpy import interp
from Model import CDA
import os, sys, argparse
from sklearn.metrics import roc_curve, auc
import pandas as pd


def main(args):

    dataset_path="../Feature"
    tprs = []
    aucs = []
    auprcs = []
    testprob = []
    y_real = []
    mean_fpr = np.linspace(0, 1, 100)

    # 读取相似度辅助信息
    dataset_feature = data_pro(dataset_path)

    # Gra_emb = read_csv('../Feature/emb_node2vec_all.csv')
    # Gra_emb_dis = Gra_emb[:88, :]  # (88, 128)
    # Gra_emb_circrna = Gra_emb[88:, :]  # (585, 128)

    all_performance = []
    for iter in range(10):
        os.system("python node2vec_emb.py")
        Gra_emb = read_csv('../Feature/emb_node2vec_all.csv')
        Gra_emb_dis = Gra_emb[:88, :]  # (88, 128)
        Gra_emb_circrna = Gra_emb[88:, :]  # (585, 128)
        for i in range(5):
            # i=(i+4)%5
            #读取正负样本，以标签的形式读取    rna索引，疾病索引，训练集标识，测试集标识，正负标识。
            circ_indices,dis_indices,train_mask,test_mask,labels=data_5fold(i)  #先取其中一折进行实验，     # train/test  1040/260

            model = CDA(gcn_layers=int(args.layer))
            model.cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

            epochs=int(args.epoch)
            model.train()
            for epoch in range(0, epochs):
                model.zero_grad()
                logits = model(dataset_feature,Gra_emb_dis,Gra_emb_circrna,circ_indices,dis_indices)
                loss =torch.nn.CrossEntropyLoss()
                loss = loss(logits[train_mask], labels[train_mask].cuda())
                loss.backward()
                optimizer.step()
                #  print("Epoch {:03d}: Loss: {:.4f}, TrainAcc {:.4}".format(epoch, loss.item(), train_acc.item()))


            model.eval()
            with torch.no_grad():
                logits = model(dataset_feature,Gra_emb_dis,Gra_emb_circrna,circ_indices,dis_indices)
                loss = torch.nn.CrossEntropyLoss()

                loss = loss(logits[test_mask], labels[test_mask].cuda())
                true_y = labels[test_mask].cpu().numpy()

                prob = F.softmax(logits[test_mask], dim=1)[:,1].cpu().numpy()
                pred_y = torch.max(F.softmax(logits[test_mask],dim=1), 1)[1].int()
                testprob.append(prob)

                fpr, tpr, thresholds = roc_curve(true_y, prob)
                tprs.append(interp(mean_fpr, fpr, tpr))
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                precision, recall, thresholds = metrics.precision_recall_curve(true_y, prob)
                y_real.append(true_y)
                auprc = auc(recall, precision)
                auprcs.append(auprc)

                acc, precision, sensitivity, specificity, MCC, f1_score,tp,fn,tn,fp = metric_calculate_performace(len(true_y), pred_y,true_y)

                # print('Test : loss: {:.4f}, Accu {:.4f}, Precision {}, Sn {:.4f}, Sp {:.4f}, MCC {:.4f}, F1_score {:.4f}, AUC {:.4f}, AUPR {:.4f}'
                #       .format(loss.item(),acc, precision, sensitivity, specificity, MCC, f1_score,roc_auc,auprc))
                all_performance.append([acc, precision, sensitivity, specificity, MCC,f1_score, roc_auc, auprc])
                model.train()
                # print(epoch,roc_auc)
            print("迭代",str(iter)+'*'+str(i),np.array(all_performance).shape)


    #   保存绘制ROC曲线和PR曲线的值
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    dataframe = pd.DataFrame({'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr})
    dataframe.to_csv("../results/ROC_MVGNCDA.csv", index=False, sep=',')


    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(testprob)
    mean_precision, mean_recall, _ = metrics.precision_recall_curve(y_real, y_proba)
    # mean_aupr = auc(mean_recall, mean_precision)

    dataframe = pd.DataFrame({'mean_precision': mean_precision, 'mean_recall': mean_recall})
    dataframe.to_csv("../results/PR_MVGNCDA.csv", index=False, sep=',')



    Mean_Result = np.mean(np.array(all_performance), axis=0)
    std_Result = np.std(np.array(all_performance), axis=0)
    print(Mean_Result.shape)

    file = open('../results/model.txt', 'a')   # 追加
    file.write('\n'+str(Mean_Result[0])+'\t' +str(Mean_Result[1])+'\t' +str(Mean_Result[2])+'\t' +str(Mean_Result[3])+'\t' +str(Mean_Result[4])+'\t' +str(Mean_Result[5])+'\t'
               +str(Mean_Result[6])+'\t' +str(Mean_Result[7])+ '\n')
    file.write(str(std_Result[0])+'\t' +str(std_Result[1])+'\t' +str(std_Result[2])+'\t' +str(std_Result[3])+'\t' +str(std_Result[4])+'\t' +str(std_Result[5])+'\t'
               +str(std_Result[6])+'\t' +str(std_Result[7])+ '\n')
    file.write('\n')
    file.close()


if __name__ == "__main__":
    ## save file
    parser = argparse.ArgumentParser(description="circRNA-Disease association prediction")
    parser.add_argument("--lr", type=str, help="lr",required=True)
    parser.add_argument("--layer", type=str, help="layer",required=True)
    parser.add_argument("--epoch", type=str, help="epoch",required=True)
    args = parser.parse_args()
    main(args)
    print("I Get It")











