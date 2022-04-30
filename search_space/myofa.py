import numpy as np


class OFASearchSpace:
    def __init__(self):
        # self.num_blocks = 5  # number of blocks
        # self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
        # self.exp_ratio = [3, 4, 6]  # expansion rate
        # self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
        # self.resolution = list(range(192, 257, 4))  # input image resolutions

        self.gcn_blocks = 2  # 图坍缩前后均有一个图卷积块 gb
        self.linear_num = 2  # 预测层数
        self.direction = [1, 0]  # False di
        self.norm = [1, 0]  # True norm
        self.depth = [1, 2, 3]  # 3 d
        self.width_rate = [0.25, 0.75, 1]  # 1 wr
        self.Dropout = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]  # 0.005 drop
        self.Weightdecay = [5e-4, 8e-4, 1e-3, 4e-3]  # 5e-4 wd
        self.lr = [5e-4, 1e-3, 5e-3, 1e-2]  # 5e-4 lr
        self.out_gcn_vector = ['sum', 'mean', 'max']  # concat ogv
        self.act = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'relu6']  # relu act
        self.linear_act = ['relu', 'leaky_relu', 'relu6']

    # def sample(self, n_samples=1, nb=None, ks=None, e=None, d=None, r=None):
    #     """ randomly sample an architecture"""
    #     nb = self.num_blocks if nb is None else nb
    #     ks = self.kernel_size if ks is None else ks
    #     e = self.exp_ratio if e is None else e
    #     d = self.depth if d is None else d
    #     r = self.resolution if r is None else r
    #
    #     data = []
    #     for n in range(n_samples):
    #         # first sample layers
    #         depth = np.random.choice(d, nb, replace=True).tolist()
    #         # then sample kernel size, expansion rate and resolution
    #         kernel_size = np.random.choice(ks, size=int(np.sum(depth)), replace=True).tolist()
    #         exp_ratio = np.random.choice(e, size=int(np.sum(depth)), replace=True).tolist()
    #         resolution = int(np.random.choice(r))
    #
    #         data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r': resolution})
    #     return data

    def sample1(self, n_samples=1, gb=None, di=None, norm=None, d=None, wr=None, drop=None, wd=None, lr=None, ogv=None, act=None, linear_act=None):
        """ randomly sample an architecture"""
        gb = self.gcn_blocks if gb is None else gb
        di = self.direction if di is None else di
        norm = self.norm if norm is None else norm
        d = self.depth if d is None else d
        wr = self.width_rate if wr is None else wr
        drop = self.Dropout if drop is None else drop
        wd = self.Weightdecay if wd is None else wd
        lr = self.lr if lr is None else lr
        ogv = self.out_gcn_vector if ogv is None else ogv
        act = self.act if act is None else act
        linear_act = self.linear_act if linear_act is None else linear_act

        data = []
        for n in range(n_samples):
            # 图 属性
            direction = int(np.random.choice(di))
            # 图卷积 拉普拉斯矩阵 正则化
            normalization = int(np.random.choice(norm))
            # 每个图卷积块的卷积次数
            depth = np.random.choice(d, gb, replace=True).tolist()
            # 预测层利用率
            width_rate = np.random.choice(wr, size=self.linear_num, replace=True).tolist()
            # drop out rate
            dropout = float(np.random.choice(drop))
            # weight decay
            weight_decay = float(np.random.choice(wd))
            # learning rate
            learning_rate = float(np.random.choice(lr))
            # out gcn vector
            handle_gcn_vector = np.random.choice(ogv, size=self.gcn_blocks, replace=True).tolist()
            # activation
            activation = np.random.choice(act, size=max(self.depth)*self.gcn_blocks+self.linear_num, replace=True).tolist()
            activation_all = activation + np.random.choice(linear_act, size=self.linear_num, replace=True).tolist()
            # # first sample layers
            # depth = np.random.choice(d, nb, replace=True).tolist()
            # # then sample kernel size, expansion rate and resolution
            # kernel_size = np.random.choice(ks, size=int(np.sum(depth)), replace=True).tolist()
            # exp_ratio = np.random.choice(e, size=int(np.sum(depth)), replace=True).tolist()
            # resolution = int(np.random.choice(r))

            # data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r': resolution})
            data.append({'di': direction, 'norm': normalization, 'd': depth, 'wr': width_rate, 'drop': dropout,
                         'wd': weight_decay, 'lr': learning_rate, 'ogv': handle_gcn_vector, 'act': activation_all})

        return data

    # def initialize(self, n_doe):
    #     # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
    #     data = [
    #         self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
    #                     d=[min(self.depth)], r=[min(self.resolution)])[0],
    #         self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
    #                     d=[max(self.depth)], r=[max(self.resolution)])[0]
    #     ]
    #     data.extend(self.sample(n_samples=n_doe - 2))
    #     return data

    def initialize1(self, n_doe):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = [
            self.sample1(1, d=[max(self.depth)], wr=[max(self.width_rate)],
                         di=[0], norm=[1], drop=[0.005], wd=[5e-4], lr=[5e-4], ogv=['sum'], act=['relu'])[0],
            self.sample1(1, d=[min(self.depth)], wr=[min(self.width_rate)])[0],
        ]
        data.extend(self.sample1(n_samples=n_doe - 2))
        return data


    def pad_zero(self, x, depth):
        # pad zeros to make bit-string of equal length
        new_x, counter = [], 0
        for d in depth:
            for _ in range(d):
                new_x.append(x[counter])
                counter += 1
            if d < max(self.depth):
                new_x += [0] * (max(self.depth) - d)
        return new_x

    def encode(self, config):
        # encode config ({'ks': , 'd': , etc}) to integer bit-string [1, 0, 2, 1, ...]
        x = []
        # 是否为有向图
        direction = [np.argwhere(config['di'] == np.array(self.direction))]
        # 是否为图坍缩是使用正则化
        norm = [np.argwhere(config['norm'] == np.array(self.norm))]
        # 两个图卷积块的卷积层数
        depth = [np.argwhere(_x == np.array(self.depth))[0, 0] for _x in config['d']]
        # 预测层 width rate
        widthrate = [np.argwhere(_x == np.array(self.width_rate))[0, 0] for _x in config['wr']]
        # 网络 dropout
        dropout = [np.argwhere(config['drop'] == np.array(self.Dropout))]
        # 网络 wight decay
        weightdecay = [np.argwhere(config['wd'] == np.array(self.Weightdecay))]
        # 网络 learning rate
        lr = [np.argwhere(config['lr'] == np.array(self.lr))]
        # 每个图网络块内 每次卷积输出的特征向量的方式
        ogv = [np.argwhere(_x == np.array(self.out_gcn_vector))[0, 0] for _x in config['ogv']]
        # 各个网络层的激活函数
        act = [np.argwhere(_x == np.array(self.act))[0, 0] for _x in config['act']]

        x += direction
        x += norm
        x += depth
        x += widthrate
        x += dropout
        x += weightdecay
        x += lr
        x += ogv
        x += act

        return x

    def decode(self, x):
        """
        remove un-expressed part of the chromosome
        assumes x = [block1, block2, ..., block5, resolution, width_mult];
        block_i = [depth, kernel_size, exp_rate]
        """
        depth, kernel_size, exp_rate = [], [], []
        for i in range(0, len(x) - 2, 9):
            depth.append(self.depth[x[i]])
            kernel_size.extend(np.array(self.kernel_size)[x[i + 1:i + 1 + self.depth[x[i]]]].tolist())
            exp_rate.extend(np.array(self.exp_ratio)[x[i + 5:i + 5 + self.depth[x[i]]]].tolist())
        return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 'r': self.resolution[x[-1]]}
