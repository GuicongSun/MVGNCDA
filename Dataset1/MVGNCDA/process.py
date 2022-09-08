# -*- coding: utf-8 -*-
import os

# # data preproces
# os.system("python Data_load.py")
# os.system("python Save_data.py")

# #  Graph embedding (node2vec)
# python node2vec_emb.py --p  0.5 --q    4

# # Model Training
#   python main.py  --lr 0.01 --layer 2 --epoch 140





#   手动修改保存文件名
os.system("python main_G.py  --lr 0.01 --layer 3 --epoch 100")
os.system("python main_N.py  --lr 0.01 --layer 3 --epoch 100")
# os.system("python main.py  --lr 0.01 --layer 3 --epoch 100")



#   python main.py  --lr 0.01 --layer 1 --epoch 2



