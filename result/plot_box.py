# Import libraries
import matplotlib.pyplot as plt
import numpy as np


# Creating dataset
np.random.seed(10)

data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data_1, data_2, data_3, data_4]

fig = plt.figure(figsize =(10, 7))

# Creating axes instance
ax = plt.gca()

# Creating plot
bp = ax.boxplot(data, sym='r+')
print(f'bp: {bp}')
for key in bp:
    print(f'{key}: {[item.get_ydata() for item in bp[key]]}\n')

print(f'medians: {bp["medians"]}')
medians = []
print("medians")
for i in bp["medians"]:
    print(i.get_ydata().tolist()[0])
    medians.append(i.get_ydata().tolist()[0])
x_medians = [i for i in range(1, len(medians)+1)]
plt.plot(x_medians, medians, ls='--', marker='^', markersize=10, label='median result')

maximums = []
print("caps")
for index, i in enumerate(bp['caps']):
    if index % 2 == 1:
        print(i.get_ydata().tolist()[0])
        maximums.append(i.get_ydata().tolist()[0])
plt.plot(x_medians, maximums, ls='-.', marker='x', markersize=10, label='best result')
plt.legend()
# show plot
plt.show()