import random
from myofa.can_codebase.data_providers.graph_sampler import GraphSampler

import torch
import numpy as np
from logger.logger import logger


# 该函数 为 单独训练图网络准备数据
def prepare_data(graphs, graphs_list, args, test_graphs=None, max_nodes=0, seed=0):
    '''
    :param graphs: 原始图
    :param graphs_list: 坍缩图
    :param args:
    :param test_graphs:
    :param max_nodes:
    :param seed:
    :return:
    '''
    zip_list = list(zip(graphs, graphs_list))
    random.Random(seed).shuffle(zip_list)
    graphs, graphs_list = zip(*zip_list)
    logger.info(f'Test ratio: {args.test_ratio}')
    logger.info(f'Train ratio: {args.train_ratio}')
    test_graphs_list = []

    if test_graphs is None:  # 有训练集 验证集 测试集
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
        train_graphs_list = graphs_list[:train_idx]
        val_graphs_list = graphs_list[train_idx: test_idx]
        test_graphs_list = graphs_list[test_idx:]
    else:  # 有训练集 验证集
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        train_graphs_list = graphs_list[:train_idx]
        val_graphs = graphs[train_idx:]
        val_graphs_list = graphs_list[train_idx:]

    # 输出信息到log文件

    logger.info(f'Num training graphs: {len(train_graphs)}; Num validation graphs: {len(val_graphs)}; '
                f'Num testing graphs: {len(test_graphs)}\n')
    logger.info(f'Number of graphs: {len(graphs)}')
    logger.info(f'Number of edges: {sum([G.number_of_edges() for G in graphs])}')
    logger.info(f'Max, avg, std of graph size: {max([G.number_of_nodes() for G in graphs])},' 
                f'{(np.mean([G.number_of_nodes() for G in graphs])):.2f},' 
                f'{np.std([G.number_of_nodes() for G in graphs]):.2f}')

    test_dataset_loader = []

    dataset_sampler = GraphSampler(train_graphs, train_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,
                                   normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type, norm=args.norm)

    sample_trainx0 = dataset_sampler[0]

    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, val_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,
                                   normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type, norm=args.norm)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)
    if len(test_graphs) > 0:
        dataset_sampler = GraphSampler(test_graphs, test_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,
                                       normalize=False, max_num_nodes=max_nodes,
                                       features=args.feature_type, norm=args.norm)
        test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return sample_trainx0, train_dataset_loader, val_dataset_loader, test_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim
