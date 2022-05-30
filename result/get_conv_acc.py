import os
import re
import matplotlib.pyplot as plt
import matplotlib

from utils import handle_log
log_dir = '../../../experiment/evo_can_conv/p0_20220509_192108/00000_log_offspring'
# adj_log_file_list = handle_log.adj_log_order(log_dir)
# for log_file in adj_log_file_list:
#     print(log_file)
# 记录每代的个体数
population_num_gen = []
# 取出初代 每个log文件的 最终精度
p0_acc_list = []
for log_file in os.listdir(log_dir):
    with open(f'{log_dir}/{log_file}', 'r') as f:
        for line in f:
            if 'best_acc' in line:
                p0_acc_list.append(float(re.findall(r'best_acc = (.*)', line)[0]))
                break
print(p0_acc_list)
print(len(p0_acc_list))
population_num_gen.append(len(p0_acc_list))


# 取出所有子代的精度
log_dir = '../../../experiment/evo_can_conv/20220518_164354_info'
for i in range(1, 31):
    p_log_dir = f'{log_dir}/{str(i).zfill(5)}_log_offspring'
    exec(f'p{i}_acc_list = []')
    for log_file in os.listdir(p_log_dir):
        with open(f'{p_log_dir}/{log_file}', 'r') as f:
            for line in f:
                if 'best_acc' in line:
                    exec(f'p{i}_acc_list.append(float(re.findall(r"best_acc = (.*)", line)[0]))')
                    break
    # 读取完 本代的 精度
    exec(f'population_num_gen.append(len(p{i}_acc_list))')
plt.figure(figsize=(18, 10))
ax = plt.gca()
# ax.axes.xaxis.set_ticks([])

matplotlib.rcParams['font.sans-serif'] = ['SimSun']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

plt.xlabel('演化代数', fontsize=25)
plt.ylabel('入侵检测精度', fontsize=20)
# plt.title('CNN进化结果', fontsize=20)
p0_x = [i for i in range(len(p0_acc_list))]
plt.plot(p0_x, p0_acc_list, 'o', alpha=0.4, markersize=5)
print(f'p0_x: {p0_x}')
# plt.show()
# r red b blue g green y yellow c cyan m 粉紫
color = ['b', 'g', 'y', 'c', 'm']
for i in range(1, 31):
    # exec(f'p{i}_x = [j+len(p{i-1}_x)*{i} for j in range(len(p{i}_acc_list))]')
    exec(f'p{i}_x = [j+sum(population_num_gen[:i]) for j in range(len(p{i}_acc_list))]')

    exec(f'print("pi_x: ", len(p{i}_x), p{i}_x)')
    exec(f'plt.plot(p{i}_x, p{i}_acc_list, "{color[i%(len(color))]}o", alpha=0.4, markersize=5)')

x_tick = [i for i in range(32)]
print(f'xticks: {p30_x[-1]+1}, {len(x_tick)}')
plt.xticks([i for i in range(0, p30_x[-1]+1, 26)], x_tick)
plt.xticks(fontproperties='Times New Roman', size=15)
plt.yticks(fontproperties='Times New Roman', size=15)
plt.xlim(-5, sum(population_num_gen)+5)
# plt.ylim(0.5, max(p30_acc_list)+0.1)
# plt.show()
plt.savefig('cnn_evo_chinese3.png', bbox_inches='tight',pad_inches=0.05, dpi=300)
    # break
