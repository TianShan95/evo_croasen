import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import time
from torchprofile import profile_macs
import torch.utils.data as Data

csign = 10e-15
since = time.time()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.net_lstm1 = nn.LSTM(128, 512)
        self.net_lstm2 = nn.LSTM(512, 512)
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(512, 64)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x, (hn, cn) = self.net_lstm1(x)
        x = self.tanh(x)

        x, (hn, cn) = self.net_lstm2(x, (hn, cn))
        x = self.tanh(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        return x

    # def loss(self, b, b_):
    #     loss_bit = []
    #     # for i in range(len(b)):
    #     #     loss_bit.append(-(b[i]*math.log(b_[i]+csign) + (1-b[i])*math.log(1-b_[i]+csign)))
    #     #
    #     # return torch.max(torch.tensor(loss_bit))  # torch.tensor(max(loss_bit))
    #
    #     for i in range(len(b)):
    #         loss_bit.append(nn.CrossEntropyLoss(b[i], b_[i]))
    #     return torch.max(torch.tensor(loss_bit))

# net = nn.Sequential(
#     nn.Linear(64, 128),
#     nn.Tanh(),
#     nn.Linear(128, 128),
#     nn.Tanh(),
#     nn.LSTM(128, 512),
#     # nn.Tanh(),
#     # nn.LSTM(512, 512),
#     # nn.Tanh(),
#     # nn.Linear(512, 64),
#     # nn.Sigmoid()
# )


import os
print(os.listdir('../../../../'))
bash_dir = '../../../../'
data_dir = f'{bash_dir}data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_1/D_1_220_feature_31412.npz'
data = np.load(data_dir)['arr_0']
# data = np.array(data)
# print(np.array(data))
label_dir = f'{bash_dir}data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_1/D_1_220_label_31412.npz'
label = np.load(label_dir)['arr_0']

net = Net()
net.load_state_dict(torch.load(f'{bash_dir}experiment/lstm/D_0_220_feature_para.pth'))
net.eval()

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')

x = torch.unsqueeze(torch.Tensor(data[:-1]), dim=1)
y = torch.Tensor(data[1:])

profile_macs(net, x[0])

# dataset = TensorDataset(torch.Tensor(data).to(device))
#
# train_loader = Data.DataLoader(
#             dataset=dataset,  # 训练的数据
#             batch_size=1,
#             shuffle=False,  # 打乱
#             num_workers=0,
#         )


# 8 epoch cuda  442.9758641719818
# 8 epoch cpu   971.3767235279083
# 200 epoch cuda consume time: 11857.058079481125
net.to(device)
loss_normal_list = []
loss_attack_list = []

# optimize = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
correct = 0
for idx in range(len(data)-1):
    in_put = Variable(torch.unsqueeze(x[idx], dim=1)).to(device)
    target = Variable(y[idx]).to(device)
    out_put = net(in_put)
    out_put = torch.squeeze(out_put)
    loss = -torch.max(target*torch.log(out_put+csign) + (1-target)*torch.log(1-out_put+csign))
    # if label[idx][0] == 0:
    #     loss_normal_list.append(loss.item())
    # else:
    #     loss_attack_list.append(loss.item())
    # print(f'label: {label[idx]}, maxloss: {loss.item()}')
    # max_normal_loss: 0.0016090695280581713
    if loss > 0.0016090695280581713:
        if label[idx][0] == 1:
            correct += 1
    else:
        if label[idx][0] == 0:
            correct += 1

# print(f'max_normal_loss: {max(loss_normal_list)}')
# print(f'min_attack_loss: {min(loss_attack_list)}')
# print(f'len(loss_normal_list): {len(loss_normal_list)}')
# print(f'len(loss_attack_list): {len(loss_attack_list)}')
print(f'acc: {correct/len(data)}')
print(f'len(data): {len(data)}')
print(f'len(label): {len(label)}')
print(f'{device} consume time: {time.time()-since}')
