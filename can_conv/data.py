import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as Data


class MyData:
    def __init__(self):

        x = np.load("../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/Conv_CAN/D_1_2/D_1_2_feature_58475.npz",)['arr_0']
        y = np.load("../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/Conv_CAN/D_1_2/D_1_2_label_58475.npz")['arr_0']

        train_size = int(len(x)*0.64)
        valid_size = int(len(x)*0.70)
        test_size = int(len(x)*0.2)

        self.shuffle = True
        self.x = torch.unsqueeze(torch.Tensor(x), dim=1).float()
        self.y = torch.Tensor(y).long()

        train_data = TensorDataset(self.x[:train_size], self.y[:train_size])
        self.train_loader = Data.DataLoader(
            dataset=train_data,  # 训练的数据
            batch_size=32,
            shuffle=self.shuffle,  # 打乱
            num_workers=8,
        )

        self.x_valid_data = self.x[train_size:valid_size]
        self.y_valid_data = self.y[train_size:valid_size]

    def print_shuffle(self):
        # print("数据是否打乱", self.shuffle)

        return self.shuffle



if __name__ == '__main__':
    p = MyData()
    p.print_shuffle()

    print(len(p.train_loader))
