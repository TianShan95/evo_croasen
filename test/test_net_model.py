import torch
import myofa

model = torch.load('../../experiment/evo_croasen_20220504_143455/iter_30/net_3/checkpoint/model_best.pth.tar')
# # open('../../experiment/evo_croasen_20220506/iter_30/net_3/checkpoint/model_best.pth.tar')
print(model['state_dict'])
print(model['model'])
# import os
# print(os.listdir('../../experiment/evo_croasen_20220504_143455/iter_30/net_3/checkpoint'))