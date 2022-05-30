import os
import re
import matplotlib.pyplot as plt
import matplotlib
import utils.handle_log as handle_log

# 取出所有子代的精度
p_log_dir = '../../../experiment/evo_can_conv/p0_20220509_192108/00000_log_offspring'
log_file_list = handle_log.adj_log_order(p_log_dir)
p_acc_list = []
for log_file in log_file_list:
    with open(f'{p_log_dir}/{log_file}', 'r') as f:
        for line in f:
            if 'best_acc' in line:
                p_acc_list.append(float(re.findall(r"best_acc = (.*)", line)[0]))
                break
# 读取完 本代的 精度
plt.figure(figsize=(18, 10))
ax = plt.gca()
# ax.axes.xaxis.set_ticks([])

matplotlib.rcParams['font.sans-serif'] = ['SimSun']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

plt.xlabel('演化代数', fontsize=25)
plt.ylabel('入侵检测精度', fontsize=20)
# plt.title('CNN进化结果', fontsize=20)
# p_x = [i for i in range(len(p_acc_list))]
# plt.plot(p_x, p_acc_list, alpha=0.4, markersize=5)
# plt.show()

print(sorted(p_acc_list))
print(len(p_acc_list))
print(p_acc_list.index(max(p_acc_list)))
