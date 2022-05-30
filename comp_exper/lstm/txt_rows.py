import os
a= 0
file_dir = '../../../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_1/D_1_220_class.txt'
with open(file_dir, 'r') as f:
    for line in f:
        a += 1
print(f'{os.path.basename(file_dir)} rows: {a}')