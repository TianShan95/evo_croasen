import torch
import myofa

# model = torch.load('/home/workspace/_share/paper_01_souce_code/experiment/evo_croasen_20220504_143455/iter_30/net_3/checkpoint/model_best.pth.tar')

# model = torch.load('/home/workspace/_share/paper_01_souce_code/experiment/evo_croasen/super_net/super_model_best.pth.tar')

model = torch.load('/home/workspace/_share/paper_01_souce_code/experiment/evo_croasen/checkpoint/model_best.pth.tar')


# /experiment/evo_croasen/super_net/super_model_best.pth.tar
# # open('../../experiment/evo_croasen_20220506/iter_30/net_3/checkpoint/model_best.pth.tar')
for item in model:
    print(item)
print(model['state_dict'])
# print(model['model'])
# import os
# print(os.listdir('../../experiment/evo_croasen_20220504_143455/iter_30/net_3/checkpoint'))