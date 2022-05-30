import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import time
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

net = Net()

import os
print(os.listdir('../../../../'))
bash_dir = '../../../../'
data_dir = f'{bash_dir}data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_0/D_0_220_feature.npz'
data = np.load(data_dir)['arr_0']
# data = np.array(data)
# print(np.array(data))
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')

x = torch.unsqueeze(torch.Tensor(data[:-1]), dim=1)
y = torch.Tensor(data[1:])


# dataset = TensorDataset(torch.Tensor(data).to(device))
#
# train_loader = Data.DataLoader(
#             dataset=dataset,  # 训练的数据
#             batch_size=1,
#             shuffle=False,  # 打乱
#             num_workers=0,
#         )
epochs = 200

# 8 epoch cuda  442.9758641719818
# 8 epoch cpu   971.3767235279083
# 200 epoch cuda consume time: 11857.058079481125
net.to(device)
optimize = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
for epoch in range(epochs):
    for idx in range(len(data)-1):
        optimize.zero_grad()

        net.train()
        in_put = Variable(torch.unsqueeze(x[idx], dim=1)).to(device)
        label = Variable(y[idx]).to(device)
        out_put = net(in_put)
        out_put = torch.squeeze(out_put)
        loss = -torch.mean(label*torch.log(out_put+csign) + (1-label)*torch.log(1-out_put+csign))
        loss.backward()
        optimize.step()
        print(f'epoch: {epoch}, loss: {loss.item()}')

print(f'{device} consume time: {time.time()-since}')

os.makedirs(f'{bash_dir}/experiment/lstm/', exist_ok=True)
torch.save(net, f'{bash_dir}/experiment/lstm/{os.path.splitext(os.path.basename(data_dir))[0]}.pth')
torch.save(net.state_dict(), f'{bash_dir}/experiment/lstm/{os.path.splitext(os.path.basename(data_dir))[0]}_para.pth')