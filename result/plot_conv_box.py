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
all_data = []
p0_acc_list = []
for log_file in os.listdir(log_dir):
    with open(f'{log_dir}/{log_file}', 'r') as f:
        for line in f:
            if 'best_acc' in line:
                p0_acc_list.append(float(re.findall(r'best_acc = (.*)', line)[0]))
                break
print(p0_acc_list)
all_data.append(p0_acc_list)
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
    exec(f'all_data.append(p{i}_acc_list)')
    # 读取完 本代的 精度
    exec(f'population_num_gen.append(len(p{i}_acc_list))')
plt.figure(figsize=(18, 10))
plt.xlabel('The number of generations', fontsize=25)
plt.ylabel('Classification accuracy', fontsize=20)
plt.xticks(fontproperties='Times New Roman', size=15)
plt.yticks(fontproperties='Times New Roman', size=15)
ax = plt.gca()
plt.rcParams['boxplot.flierprops.markersize'] = 10
bp = ax.boxplot(all_data, sym='r+')
medians = []
# print("medians")
for i in bp["medians"]:
    # print(i.get_ydata().tolist()[0])
    medians.append(i.get_ydata().tolist()[0])
x_medians = [i for i in range(1, len(medians)+1)]
plt.plot(x_medians, medians, ls='--', marker='^', markersize=10, label='median result')

maximums = []
# print("caps")
for index, i in enumerate(bp['caps']):
    if index % 2 == 1:
        # print(i.get_ydata().tolist()[0])
        maximums.append(i.get_ydata().tolist()[0])
plt.plot(x_medians, maximums, ls='-.', marker='x', markersize=10, label='best result')
plt.legend()


# plt.show()
plt.savefig('cnn_evo_eng_01.png', bbox_inches='tight',pad_inches=0.05, dpi=300)

