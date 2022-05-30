# import numpy as np
#
# a = np.array([1, 2, 3])
# print(a)
import os
bash_dir = '../../../../'
data_dir = f'{bash_dir}data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_0/D_0_220_feature.npz'
print(os.path.splitext(os.path.basename(data_dir))[0])
os.makedirs(f'{bash_dir}/experiment/lstm/', exist_ok=True)
