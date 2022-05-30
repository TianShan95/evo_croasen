import numpy as np

fr = open('../../../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_1/D_1_220_class.txt')
# 按行遍历读取文件
count = 0
label_mat = []
for line in fr.readlines():
    # count += 1
    # print(count)
    # 每一行先去掉回车换行符，再以Tab键为元素之间的分割符号，把每一行分割成若干元素
    lineArr = line.strip()
    lineArr = [int(x) for x in lineArr]
    label_mat.append(lineArr)  # 存入一行

label_np = np.array(label_mat)
np.savez(f"../../../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_1/D_1_220_label_{len(label_np)}.npz", label_mat)