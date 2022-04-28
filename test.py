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
import torch
df = open('../experiment/evo_croasen/checkpoint/model_best.pth.tar', 'rb')
data = torch.load(df, encoding='bytes')
print(data)