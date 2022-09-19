# -*- coding:utf-8 -*-
import csv
import numpy as np
import pandas as pd
from ge import Node2Vec
import networkx as nx
import os, sys, argparse

def node2vec_emb():
    #   参数寻优得到的结果
    pp=2
    qq=0.25

    G = nx.read_edgelist('../Feature/Edgelist_all.txt', create_using=nx.DiGraph(), nodetype=None,data=[('weight', int)])
    model = Node2Vec(G, walk_length=10, num_walks=80, p=pp, q=qq, workers=1, use_rejection_sampling=0)
    model.train(embed_size=64, window_size=10, iter=1)
    embeddings = model.get_embeddings()

    ### len(embeddings)=88 + 585
    # 先88的，再585的，也就process transit是说：先是疾病再是circRNA
    Gemb_deepwalk = np.zeros([len(embeddings), 64])  # num_dimension num_feature

    for emb in range(len(embeddings)):
        for i in range(64):
            Gemb_deepwalk[emb][i] = embeddings[str(emb)][i]

    result = pd.DataFrame(Gemb_deepwalk)
    result.to_csv('../Feature/emb_node2vec_all.csv', header=False, index=False)




if __name__ == "__main__":
    node2vec_emb()









