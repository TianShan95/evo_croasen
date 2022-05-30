import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import TensorDataset
import torch.utils.data as Data
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torchprofile import profile_macs


net = nn.Sequential(
    nn.Linear(64, 128),
    nn.Sigmoid(),
    nn.Linear(128, 512),
    nn.Sigmoid(),
    nn.Linear(512, 256),
    nn.Sigmoid(),
    nn.Linear(256, 32),
    nn.Sigmoid(),
    nn.Linear(32, 2)
)

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')
net.to(device)

bash_dir = '../../../../'
data_dir = f'{bash_dir}data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_1/D_1_220_feature_31412.npz'
data = np.load(data_dir)['arr_0']

label_dir = f'{bash_dir}data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_1/D_1_220_label_31412.npz'
label = np.load(label_dir)['arr_0']

seed = 42
zip_list = list(zip(data, label))
random.Random(seed).shuffle(zip_list)
data, label = zip(*zip_list)

train_idx = int(len(data) * 0.8)
train_data = data[:train_idx]
train_label = label[:train_idx]
val_data = data[train_idx:]
val_label = label[train_idx:]

train_data = torch.Tensor(train_data).float()
train_label = torch.Tensor(train_label).long()

val_data = torch.Tensor(val_data).float()
val_label = torch.Tensor(val_label).long()

train_dataset = TensorDataset(train_data, train_label)
train_loader = Data.DataLoader(
            dataset=train_dataset,  # 训练的数据
            batch_size=32,
            shuffle=True,  # 打乱
            num_workers=4,
        )
epochs = 8
optimize = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
loss_func = nn.CrossEntropyLoss().to(device)  # 二分类损失函数
train_loss_list = []

for epoch in range(epochs):
    net.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        optimize.zero_grad()
        b_x = Variable(b_x).to(device)
        b_y = Variable(torch.squeeze(b_y)).to(device)
        output = net(b_x)  # net 在训练 batch 上的输出
        train_loss = loss_func(output, b_y)  # 二分类交叉熵损失函数
        train_loss_list.append(train_loss.item())
        train_loss.backward()
        optimize.step()
        niter = epoch * len(train_loader) + step + 1
        if niter % 50 == 0:
            print(f'loss: {train_loss.item()}')
    net.eval()
    x_valid = val_data.to(device)
    output_pre = net(x_valid)
    _, pre_lab = torch.max(output_pre, 1)
    # print('pre_lab: ', pre_lab)
    pre_lab = pre_lab.cpu()
    y_valid = torch.squeeze(val_label)
    valid_accuracy = accuracy_score(y_valid, pre_lab)  # 临时 变量 记录 本次的 验证精度
    print(f'epoch: {epoch} acc: {valid_accuracy}')

# plot loss
plot_loss = [train_loss_list[i] for i in range(0, len(train_loss_list), 500)]
x = [i for i, v in enumerate(plot_loss)]

plt.plot(x, plot_loss)
print('showing')
plt.show()