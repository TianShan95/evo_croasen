# import matplotlib.pyplot as plt
#
# plt.plot([1, 4, 2, 8])
# plt.show()


import os
# base_dir = '../.tmp/iter_0'
# base_dir = os.path.dirname(base_dir)
# print(base_dir)

# iter_0 = [1, 2, 3, 4]
# exec(f'for file in iter_0:'
#      f' print(file)')

# base_dir = '/home/workspace/_share/paper_01_souce_code/GCN/evo_croasen/.tmp'
# print(os.listdir(base_dir))

# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

x = np.linspace(-10, 10, 200)
y = x
plt.plot(x, y)

# 设置matplotlib正常显示中文和负号
fontP = font_manager.FontProperties()
fontP.set_family('sans-serif')

plt.xlabel("横轴/单位", fontproperties=fontP)
plt.ylabel("纵轴/单位", fontproperties=fontP)
plt.title("标题", fontproperties=fontP)
plt.show()
