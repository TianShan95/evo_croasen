import pandas
import time

df = pandas.read_csv('../../../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_1.csv')
# print(df.head())


id_list = df.get("Arbitration_ID").values
data_list = df.get('Data').values
label_list = df.get('Class').values
print(f'len(ID) {len(id_list)}, len(data) {len(data_list)}, len(label) {len(label_list)}')
time.sleep(1)
# print(len(id_list))
# print(len(data_list))
import os
print(os.listdir("../../../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can"))
# with open('../../../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_0/D_1_220_data.txt', 'w+') as f:

    # 取 特定CANID 的数据段
    # for index, id in enumerate(id_list):
    #     if id == '220':
    #         print(data_list[index])
    #         # data = data_list[index].replace(' ', '')
    #         # print(data)
    #         f.write(f'{(bin(int(data_list[index].replace(" ", ""), 16)))[2:].zfill(64)}\n')
    #         # break

# 0 表示Normal 1 表示 Attack
with open('../../../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/lstm_can/D_0/D_1_220_class.txt','w+') as f:
    # 取 特定CANID 的 Attack和Normal标签
    for index, id in enumerate(id_list):
        if id == '220':
            # print(label_list[index])
            # print(label_list[index])
            if label_list[index] == 'Normal':
                print('ok')
                f.write(f'0\n')
            else:
                print('no')
                f.write(f'1\n')