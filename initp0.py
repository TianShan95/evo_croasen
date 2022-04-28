import random


def initp(k):

    p0_list = []  # 初始种群列表
    n_gene = 10  # 基因的个数

    for i in range(k):
        # gene0 有向图/无向图
        gen0 = random.randint(0, 1)  # 是左闭右闭的区间
        # gen1 坍缩前卷积层个数
        gen1 = random.randint(0, 2)
        # gen2 坍缩后卷积层个数
        gen2 = random.randint(0, 2)
        # gen3 拉普拉斯矩阵是否正则化
        gen3 = random.randint(0, 1)
        # gen4 卷积后特征向量输入预测层的方式
        gen4 = random.randint(0, 4)
        # gen5 预测层激活函数类型
        gen5 = random.randint(0, 4)
        # gen6 隐藏层神经元个数
        gen6 = random.randint(0, 4)
        # gen7 DropOut
        gen7 = random.randint(0, 5)
        # gen8 Weight Decay Rate
        gen8 = random.randint(0, 3)
        # gen9 Learning Rate
        gen9 = random.randint(0, 3)

        p0_list.append([])
        for g in range(n_gene):
            exec(f'p0_list[i].append(gen{g})')

    return p0_list


if __name__ == '__main__':
    n = 100
    initp(n)