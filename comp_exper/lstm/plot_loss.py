import re

import matplotlib.pyplot as plt
loss_list = []
with open('/Users/aaron/Downloads/0518_1638.log', 'r') as f:
    for line in f:
        # print(line)
        if 'epoch: ' in line:
            loss = float(re.findall(r'loss: (.*)', line)[0])
            loss_list.append(loss)
print('get over')

loss_plt_list = [loss_list[i] for i in range(0, len(loss_list), 500)]
print(len(loss_plt_list))

x = [i for i, v in enumerate(loss_plt_list)]
print(x)
print(loss_plt_list)
# print(loss_list[:500])
plt.plot(x, loss_plt_list)
# print('ploting')
# plt.savefig('plt.png', dpi=300, bbox_inches='tight')
# print('saved')
plt.show()