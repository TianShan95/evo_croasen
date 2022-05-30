import numpy as np
import copy

f = open('../../../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_1/D_1_220_data.txt')
feature_mat = []
feature_r = []
for line in f.readlines():
    lineArr = list(line.strip())
    for i in lineArr:  # 添加一行 29 个二进制位
        feature_r.append(int(i))  # 共有 29 个二进制位
    feature_mat.append(copy.deepcopy(feature_r))
    feature_r.clear()

feature_mat = np.array(feature_mat)

np.savez(f"../../../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/"
         f"lstm_can/D_1/D_1_220_feature_{len(feature_mat)}.npz", feature_mat)

