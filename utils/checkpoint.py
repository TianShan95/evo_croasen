import torch

f = open('../../experiment/evo_croasen/checkpoint/checkpoint.pth.tar','rb')

data = torch.load(f, encoding='Bytes')
print(data['best_acc'])
