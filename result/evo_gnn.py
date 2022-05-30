import os
import re
import json
import matplotlib.pyplot as plt


base_dir = '../.tmp/iter_0'
# base_dir = '/home/workspace/_share/paper_01_souce_code/GCN/evo_croasen/.tmp/iter_0'

files = os.listdir(base_dir)
iter_num = []
for file in files:
    # print(re.match(r'net_\d+_stats', file))
    if re.match(r'net_\d+_stats', file):
        # print(file)
        iter_num.append(file)
# print(len(iter_0))
# iter_0.sort(key=lambda x: int(re.findall(r'net_(\d+)_stats', x)[0]))
flops = []
acc_error = []
for file in iter_num:
    stats = json.load(open(os.path.join(base_dir, file)))
    flops.append(stats['flops'])
    acc_error.append(1-stats['acc'])
# plt.plot(flops, acc_error, 'bo', markersize=3, alpha=0.4)
iter_num.clear()
flops.clear()
acc_error.clear()
# plt.show()
color = ['b', 'g', 'y', 'c', 'm']

base_dir = os.path.dirname(base_dir)
for i in range(1, 31):
    base_dir_iter = os.path.join(base_dir, f'iter_{i}')
    files = os.listdir(base_dir_iter)
    for file in files:
        if re.match(r'net_\d+_stats', file):
            iter_num.append(file)
    for file in iter_num:
        stats = json.load(open(os.path.join(base_dir_iter, file)))
        flops.append(stats['flops'])
        acc_error.append(1 - stats['acc'])
    if i >= 30:
        plt.plot(flops, acc_error, 'rx', markersize=4, alpha=0.8)
    # else:
        # plt.plot(flops, acc_error, 'bo', markersize=3, alpha=0.4)
        # plt.plot(flops, acc_error, f'{color[i%len(color)]}o', markersize=3, alpha=0.4)

    iter_num.clear()
    flops.clear()
    acc_error.clear()

# plt.xlabel('计算复杂度（个体的复杂度/超网的复杂度）')
# plt.ylabel('精度误差（1-精度）')
plt.xlabel('individual flops/supernet flops')
plt.ylabel('1-acc')
# plt.title('GNN evolution')
# plt.show()
plt.savefig('gnn_evo_eng_02.png', bbox_inches='tight', pad_inches=0.01, dpi=300)

