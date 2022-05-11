# import numpy as np
#
# a = [5e-4, 1e-3, 5e-3, 1e-2]
# b = [1]
# print(np.random.choice(a))
# l = [1, 2, 3, 4, 5, 6, 7, 8]
# # for index, value in enumerate(l):
# #     a = l[index]
# #     b = a
# # print(l[3:6])
#
# a = l*3
# print(a)

# 画图
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# x = np.arange(0, 30, 0.1)
# y = []
# for t in x:
#     # y_1 = 1 / (1 + math.exp(-t))
#     y_1 = 0.5 * 0.01 * (1 + math.cos(math.pi * t / 30))
#     y.append(y_1)
# plt.plot(x, y, label="sigmoid")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.ylim(0, 0.01)
# plt.legend()
# plt.show()

# import argparse
# def a(kwargs):
#     save = kwargs.pop('save', '.tmp')
#     print(kwargs)
#     print(save)
#
# parser = argparse.ArgumentParser()
# # parser.add_argument('--save', type=str, default='.tmp111',
# #                     help='location of dir to save')
# parser.add_argument('--resume', type=str, default=None,
#                     help='resume search from a checkpoint')
# cfgs = parser.parse_args()
# a(vars(cfgs))

# import numpy as np
# x = 2
# a = [x]
# print(a, type(a))

# def a(arg, **kwargs):
#     print(arg)
#     print(kwargs)
#
# a('a', a=2)
# import torch
# a = [[1, 5, 2], [4, 2, 9]]
# b = [[3, 6, 3], [9, 3, 7]]
# c = [[4, 1, 7], [5, 9, 2]]
#
# a = torch.tensor(a)
# b = torch.tensor(b)
# c = torch.tensor(c)
# l = [a, b, c]
# # m = torch.zeros(2, 3)
# # print(m)
# m = l[0]
# for i, x in enumerate(l[:-1]):
#     m = torch.where(m > l[i+1], m, l[i+1])
#
#     print(m)

# print(a)
# print(b)
# print(torch.where(a > 1, a, b))
#
# print(torch.where(a > b, a, b))
# print(a/2)

# print(torch.zeros(2, 3, 4))
# import torch
# a = torch.rand((4, 6))
# print(a)

# a = 1
# b = 2
# c = 3
#
# l1 = [a, b, c]
# l2 = [a, b]
#
# if l2 in l1:
#     print('ok')

# 加载 pickle
#
# df = open('/Users/aaron/Hebut/征稿_图像信息安全_20211130截稿/源程序/图塌缩分类/experiment/evo_croasen/checkpoint.pth.tar', 'rb')
# data = torch.load(df, encoding="bytes")
# print(data)
# print(torch.zeros(2, 3))
# print(torch.zeros((2, 3)))
# print(torch.zeros([2, 3]))
# import torch.nn as nn
#
# net = nn.Sequential()
# module1 = nn.Dropout(0.05, inplace=True)
# net.add_module('1', module1)
# net.add_module('2', module1)
# print(module1)
# print(net)
import os
import time

import numpy as np
# a = np.array(['relu', 'softmax'])
# print(np.argwhere('softmax' == a)[0,0])
# a = []
# a += 'relu'
#
# print(a)

# for i in range(10):
#     time.sleep(1)
#     print(i)

# 取出存储的配置中的网络权重参数
# import torch
# df = open('../experiment/evo_croasen/checkpoint/model_best.pth.tar', 'rb')
# data = torch.load(df, encoding='bytes')
# print(data)

# import torch
# import torch.nn as nn
# from torchprofile import profile_macs
# from myofa.can_codebase.utils.flops_counter import profile
#
# import copy
# from ptflops import get_model_complexity_info

# net = nn.Sequential(
#     nn.Linear(40, 20),
#     nn.ReLU(),
#     nn.Linear(20, 10),
#     nn.ReLU(),
#     nn.Linear(10, 2),
#     nn.Softmax(),
# )
# x = torch.randn(64, 40)
# y = net(x)
# print(y.shape)
# flops1 = int(profile_macs(copy.deepcopy(net), x))
# print(flops1)
# flops2, _ = profile(net, (64, 40))
# print(flops2)
# flops2, params = get_model_complexity_info(net, (64, 40), as_strings=True, print_per_layer_stat=True)
# print(flops2)



# import torch
# import torch.nn as nn
# from torchprofile import profile_macs
# from myofa.can_codebase.utils.flops_counter import profile
# import torch.nn.functional as F
# import copy
# from ptflops import get_model_complexity_info
#
# class GraphConv(nn.Module):
#     def __init__( input_dim, output_dim, add_False, normalize_embedding=False,
#                  dropout=0.0, bias=True, device='cpu'):
#         super(GraphConv, .__init__()
#         add_= add_self
#         dropout = dropout
#
#         if dropout > 0.001:
#             dropout_layer = nn.Dropout(p=dropout).to(device)
#         normalize_embedding = normalize_embedding
#         input_dim = input_dim
#         output_dim = output_dim
#         weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim)).to(device)
#         if bias:
#             bias = nn.Parameter(torch.FloatTensor(output_dim)).to(device)
#         else:
#             bias = None
#
#     def forward( x, adj):
#         if dropout > 0.001:
#             x = dropout_layer(x)
#         y = torch.matmul(adj, x)
#         if add_
#             y += x
#
#         # print('weight shape: ')
#         # print(weight.shape)
#         y = torch.matmul(y, weight)
#
#         if bias is not None:
#             y = y + bias
#         if normalize_embedding:
#             y = F.normalize(y, p=2, dim=2)
#         return y
#
# gcn_net = GraphConv(81, 20)
# gcn_net1 = GraphConv(20, 20)
# net_sequential = nn.Sequential(gcn_net, gcn_net1)
#
# x = torch.randint(0, 81, (64, 81, 81)).float()
# adj = torch.randint(0, 2, (64, 81, 81)).float()
# y = net_sequential(x, adj)
# print(y.shape)
# flops1 = int(profile_macs(copy.deepcopy(net_sequential), (x, adj)))
# print(flops1)

# import torch.nn as nn
# block = nn.ModuleList()
# net_sequential1 = nn.Sequential()
# net_sequential2 = nn.Sequential()
#
# net1 = nn.Linear(60, 40)
# net2 = nn.Linear(40, 20)
#
# net_sequential1.add_module('net1', net1)
# net_sequential1.add_module('net2', net2)
# exec(f'block.append(net_sequential1)')
#
# net3 = nn.Linear(20, 10)
# net4 = nn.Linear(10, 2)
# exec(f"net_sequential2.add_module('net3', net3)")
# net_sequential2.add_module('net4', net4)
# exec(f'block.append(net_sequential2)')
# print(block)
# print(block[1].net3)
# print('okkk')


# import torch
# from torch import nn
# a = torch.tensor([0.5, 0.2], requires_grad=True)
# b = nn.Tanh()(a)
# # b = nn.Linear(1,1)(a)
# print(b)
# b *= 1
# # b = b * 1
# b.sum().backward()

import torch
import torch.nn as nn
import torch.nn.functional as F

# net = nn.Sequential(
#     nn.Linear(20, 2),
#     nn.Tanh()
# )
# linear = nn.Linear(50, 30)
# weight = linear.weight[:20, :2]

# x = torch.randn(64, 20)
# y = torch.randint(0, 2, (64,))
# nn.Tanh()(F.linear(x, weight))
# net.train()
# pred = net(x)


# class Net(nn.Module):
#     def __init__( input_dim, output_dim):
#         super().__init__()
#         input_dim = input_dim
#         output_dim = output_dim
#         linear = nn.Linear(input_dim, output_dim)
#     def forward( x):
#         weight = linear.weight[:2, :20]
#         bias = linear.bias[:2]
#         return nn.Tanh()(F.linear(x, weight.clone(), bias.clone()))
#
# net = Net(50, 30)
# net.train()
# ypred = net(x)
# loss = F.cross_entropy(ypred, y)
# loss.backward()

# a = [1, 2, 3, 4]
# b = [7, 8, 9]
# print(a+b)
# import re
# with open('iter_0_result.log') as f:
#     for line in f:
#         results = line.split('!')
# results = [x.strip() for x in results if x.strip() != '']
# results.sort(key=lambda x : int(re.findall(r'net_(\d+)_subnet.txt', x)[0]))
# # results.sort()
#
# print(results)

# img_list = [4, 2, 1, 9, 3]
# img_list.sort(key = lambda x: int(x))
# print(img_list)

# import subprocess
# print('test.py')
# subprocess.call('python test1.py & wait', shell=True)
# import re
# save_net_name = re.findall(r'\/(.*)\_subnet', '')[0]
# n_var = 19
# xu = np.ones(n_var)
# xu[2:6] = 2
# xu[6] = 5
# xu[7:9] = 3
# xu[9:11] = 2
# xu[11:17] = 4
# xu[17:] =2
# print(xu)

# a = [1, 2, 3, 4, 5, 6, 7]
# print(a[:-2])
# print(a[-2:])

# a = '/tmp/a.txt'
# print(os.path.basename(a))

# a = np.array([2, 3, 4])
# print(a*100)

# import re
# can_conv_net_dir = '../../experiment/evo_can_conv/p0_20220509_143247/00000_pkl_offspring/00_[0, 0, 2, 1, 1, 1, 0, 4]_offspring.pkl'
# gene_list_str = re.findall(r'\[(.*)\]', can_conv_net_dir)[0]
# gene_list = list(map(int, gene_list_str.split(',')))
#
# print(gene_list)
# print(type(gene_list))

# a = [21, 3, 5]
# for i in a:
#     print(int(i/2))
# print((300-28)/29)
# print((300-28)//29)
# print((300-28)%29)

# a = [
#     [1, 2],
#     [3, 4]
# ]
# b = torch.from_numpy(np.array(a))
# print(b)
# print(b[0][0])
# print(b[0][0]>b[0][1])
# if b[0][0]>b[0][1]:
#     print('ok')
# else:
#     print('ooook')
import os
a = '../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/'
print(os.path.dirname(os.path.dirname(a)))