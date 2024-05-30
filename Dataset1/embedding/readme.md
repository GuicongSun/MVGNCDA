CosineSim.py
基于Cosine生成相似度矩阵

GIPSim.py
基于GIP生成相似度矩阵

disSSim_1.py
基于第一类DAG关系生成相似度矩阵

disSSim_2.py
基于第二类DAG关系生成相似度矩阵

disSSim.py
基于两类DAG关系综合生成疾病语义相似度矩阵

circFunSim.py
基于疾病语义相似度矩阵，生成circRNA相似度矩阵

Sim_all.py
基于以上三种相似度矩阵集成为最终的相似度矩阵

node2vec_preprocess.py
基于关联矩阵生成连接边，便于后续进行node2vec图嵌入

node2vec_process_het.py
基于集成的相似度矩阵获取异构关联矩阵的c-c和d-d连接的边，

将以上两者得到的边文档，加和在一起，利用node2vec代码生成图嵌入









