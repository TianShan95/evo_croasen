from myofa.utils import random_seed
import initp0

if __name__ == '__main__':

    # init
    seed = 42
    n = 100

    # 随机种子
    random_seed.setup_seed(seed)
    # 生成初始种群
    p0_list = initp0.initp(n)
    # 把基因解构为神经网络


    # 训练所有个体
