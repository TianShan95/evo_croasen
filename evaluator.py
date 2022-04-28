#!/usr/bin/env Python
# coding=utf-8

import os
import json
import torch
import argparse
import numpy as np

import utils
# from codebase.networks import NSGANetV2
from codebase.run_manager import get_run_config
# from ofa.elastic_nn.networks import OFAMobileNetV3
from myofa.can_codebase.run_manager import RunManager
# from ofa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d

from myofa.elastic_nn.networks import OFACoarsenGcnV1
import warnings
warnings.filterwarnings('ignore')
# import warnings
# warnings.simplefilter("ignore")
#
# DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1


def parse_string_list(string):
    if isinstance(string, str):
        # convert '[5 5 5 7 7 7 3 3 7 7 7 3 3]' to [5, 5, 5, 7, 7, 7, 3, 3, 7, 7, 7, 3, 3]
        return list(map(int, string[1:-1].split()))
    else:
        return string


def pad_none(x, depth, max_depth):
    new_x, counter = [], 0
    for d in depth:
        for _ in range(d):
            new_x.append(x[counter])
            counter += 1
        if d < max_depth:
            new_x += [None] * (max_depth - d)
    return new_x


def get_net_info(net, x, args,  measure_latency=None, print_info=True, clean=False, lut=None):

    net_info = utils.get_net_info(
        net, x, args, measure_latency, print_info=print_info, clean=clean, lut=lut)

    gpu_latency, cpu_latency = None, None
    for k in net_info.keys():
        if 'gpu' in k:
            gpu_latency = np.round(net_info[k]['val'], 2)
        if 'cpu' in k:
            cpu_latency = np.round(net_info[k]['val'], 2)

    return {
        'params': np.round(net_info['params'] / 1e3, 2),
        # 'flops': np.round(net_info['flops'] / 1e6, 2),
        'gpu': gpu_latency, 'cpu': cpu_latency
    }


def validate_config(config, max_depth=4):
    depth = config['depth']

    if isinstance(depth, str): depth = parse_string_list(depth)

    assert isinstance(depth, list)

    # return {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'w': config['w']}
    # return {'ks': kernel_size, 'e': exp_ratio, 'd': depth}


class OFAEvaluator:
    """ based on OnceForAll supernet taken from https://github.com/mit-han-lab/once-for-all """
    def __init__(self,
                 n_classes=2,
                 model_path='./data/ofa_mbv3_d234_e346_k357_w1.0',
                 gcn_blocks=None, linear_num=None, direction=None, norm=None, depth=None, width_rate=None, dropout=None, weight_decay=None, out_gcn_vector=None, args=None):


        # default configurations
        self.gcn_blocks = 2 if gcn_blocks is None else gcn_blocks  # 图坍缩前后均有一个图卷积块 gb
        self.linear_num = 2 if linear_num is None else linear_num
        self.direction = [True, False]  if direction is None else direction  # False di
        self.norm = True  if norm is None else norm  # True norm
        self.depth = [3, 3]  if depth is None else depth  # 3 d
        self.width_rate = 1 if width_rate is None else width_rate  # 1 wr
        self.Dropout = 0.05 if dropout is None else dropout  # 0.05 drop
        self.Weightdecay = 5e-4  if weight_decay is None else weight_decay  # 5e-4 wd

        self.out_gcn_vector = ['sum', 'sum']  if out_gcn_vector is None else out_gcn_vector  # concat ogv
        # self.act = 'relu' if act is None else act  # relu act

        # if 'w1.0' in model_path:
        self.width_mult = [1.0]*self.linear_num
        # elif 'w1.2' in model_path:
        #     self.width_mult = 1.2
        # else:
        #     raise ValueError

        self.engine = OFACoarsenGcnV1(n_classes=n_classes, dropout_rate=self.Dropout, width_mult_list=self.width_mult,
                                      depth_list=self.depth, args=args, out_gcn_vector=self.out_gcn_vector)

        # # 保存 OFA net架构 可视化模型研究
        # torch.save(self.engine, 'OFAMobileNetV3.pth')

        # 加载模型权重参数
        # init = torch.load(model_path, map_location='cpu')['state_dict']
        # self.engine.load_weights_from_net(init)

        # torch.save(self.engine, 'OFAMobileNetV3.pth')

    def sample(self, config=None):
        """ randomly sample a sub-network """
        if config is not None:
            # config = validate_config(config)
            self.engine.set_active_subnet(depth=config['d'], width_rate=config['wr'], activation=config['act'], dropout_rate=config['drop'], out_gcn_vector=config['ogv'])
        else:
            config = self.engine.sample_active_subnet()

        subnet = self.engine.get_active_subnet(preserve_weight=True)
        return subnet, config

    @staticmethod
    def save_net_config(path, net, config_name='net.config'):
        """ dump run_config and net_config to the model_folder """
        net_save_path = os.path.join(path, config_name)
        json.dump(net.config, open(net_save_path, 'w'), indent=4)
        print('Network configs dump to %s' % net_save_path)

    @staticmethod
    def save_net(path, net, model_name):
        """ dump net weight as checkpoint """
        if isinstance(net, torch.nn.DataParallel):
            checkpoint = {'state_dict': net.module.state_dict()}
        else:
            checkpoint = {'state_dict': net.state_dict()}
        model_path = os.path.join(path, model_name)
        torch.save(checkpoint, model_path)
        print('Network model dump to %s' % model_path)

    @staticmethod
    def eval(subnet, data_path, init_lr, weight_dacay, out_gcn_vector=None, dataset='imagenet', n_epochs=0, resolution=224, trn_batch_size=128, vld_batch_size=250,
             num_workers=4, valid_size=None, is_test=True, log_dir='../experiment/evo_croasen', measure_latency=None, no_logs=False, args=None):

        lut = {'cpu': 'data/i7-8700K_lut.yaml'}

        run_config = get_run_config(init_lr=init_lr, weight_dacay=weight_dacay,
                                    dataset=dataset, data_path=data_path, image_size=resolution, n_epochs=n_epochs,
                                    train_batch_size=trn_batch_size, test_batch_size=vld_batch_size,
                                    n_worker=num_workers, valid_size=valid_size, args=args)

        # print(run_config.data_provider.n_classes)

        # for batch_idx, data in enumerate(run_config.train_loader):
            # x = run_config.train_loader
        info = get_net_info(subnet, run_config.data_provider.sample_trainx0, args, measure_latency=measure_latency,
                            print_info=False, clean=True, lut=lut)

        # set the image size. You can set any image size from 192 to 256 here
        # run_config.data_provider.assign_active_img_size(resolution)

        # if n_epochs > 0:
            # for datasets other than the one supernet was trained on (ImageNet)
            # a few epochs of training need to be applied
        # subnet.reset_classifier(
        #     last_channel=subnet.classifier_blocks.in_features,
        #     n_classes=run_config.data_provider.n_classes, dropout_rate=cfgs.drop_rate)

        run_manager = RunManager(log_dir, subnet, run_config, args=args, init=False)
        # if reset_running_statistics:
            # run_manager.reset_running_statistics(net=subnet, batch_size=vld_batch_size)
            # run_manager.reset_running_statistics(net=subnet)

        # if n_epochs > 0:
        run_manager.train(cfgs, out_gcn_vector)

        loss, acc = run_manager.validate(net=run_manager.net, is_test=is_test, no_logs=no_logs, output_graph_vector=out_gcn_vector)

        info['loss'], info['acc'] = loss, acc

        save_path = os.path.join(log_dir, 'net_stats.txt') if cfgs.save is None else cfgs.save
        if cfgs.save_config:
            import re
            save_net_name = re.findall(r'\/(.*)\_subnet', args.subnet)[0].replace("/", "_")
            OFAEvaluator.save_net_config(log_dir, subnet, "%s_config.txt" % save_net_name)
            OFAEvaluator.save_net(log_dir, subnet, "%s.init" % save_net_name)
        with open(save_path, 'w') as handle:
            json.dump(info, handle)

        print(info)


def main(args):
    """ one evaluation of a subnet or a config from a file """
    args.subnet = '.tmp/iter_0/net_0.subnet'

    mode = 'subnet'
    if args.config is not None:
        if args.init is not None:
            mode = 'config'

    print('Evaluation mode: {}'.format(mode))
    config = json.load(open(args.subnet))
    # if mode == 'config':
    #     net_config = json.load(open(args.config))
    #     subnet = NSGANetV2.build_from_config(net_config, drop_connect_rate=args.drop_connect_rate)
    #     init = torch.load(args.init, map_location='cpu')['state_dict']
    #     subnet.load_state_dict(init)
    #     subnet.classifier_blocks.dropout_rate = args.drop_rate
    #     try:
    #         resolution = net_config['resolution']
    #     except KeyError:
    #         resolution = args.resolution

    if mode == 'subnet':
        evaluator = OFAEvaluator(n_classes=args.n_classes, model_path=args.supernet_path, args=args)
        subnet, _ = evaluator.sample({'di': config['di'], 'norm': config['norm'],
                                      'wr': config['wr'], 'drop': config['drop'],
                                      'd': config['d'], 'act': config['act'], 'ogv': config['ogv']})

    else:
        raise NotImplementedError
    init_lr = config['lr']   # 5e-4 lr
    weight_dacay = config['wd']  # weight_decay
    OFAEvaluator.eval(
        subnet, init_lr=init_lr, weight_dacay=weight_dacay, out_gcn_vector=config['ogv'], log_dir=args.log_dir, data_path=args.data, dataset=args.dataset, n_epochs=args.n_epochs,
        trn_batch_size=args.trn_batch_size, vld_batch_size=args.vld_batch_size,
        num_workers=args.num_workers, valid_size=args.valid_size, is_test=args.test, measure_latency=args.latency,
        no_logs=(not args.verbose), args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/',
                        help='location of the data corpus')
    parser.add_argument('--log_dir', type=str, default='../experiment/evo_croasen/',
                        help='directory for logging')
    parser.add_argument('--dataset', type=str, default='chc',
                        help='name of the dataset (car hack challenge Dataset...)')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='number of classes for the given dataset normal or intrusion')
    parser.add_argument('--supernet_path', type=str, default='../experiment/evo_croasen/super_net/super_model_best.pth.tar',
                        help='file path to supernet weights')
    parser.add_argument('--subnet', type=str, default=None,
                        help='location of a json file of config eg: num of gcn')
    parser.add_argument('--config', type=str, default=None,
                        help='location of a json file of specific model declaration')
    parser.add_argument('--init', type=str, default=None,
                        help='location of initial weight to load')
    parser.add_argument('--trn_batch_size', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--vld_batch_size', type=int, default=256,
                        help='test batch size for inference')
    # parser.add_argument('--num_workers', type=int, default=6,
    #                     help='number of workers for data loading')
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--save', type=str, default=None,
                        help='location to save the evaluated metrics')
    # parser.add_argument('--resolution', type=int, default=224,
    #                     help='input resolution (192 -> 256)')
    parser.add_argument('--valid_size', type=int, default=None,
                        help='validation set size, randomly sampled from training set')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluation performance on testing set')
    parser.add_argument('--latency', type=str, default=None,
                        help='latency measurement settings (gpu64#cpu)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='whether to display evaluation progress')
    parser.add_argument('--reset_running_statistics', action='store_true', default=False,
                        help='reset the running mean / std of BN')
    # parser.add_argument('--drop_rate', type=float, default=0.2,
    #                     help='dropout rate')
    # parser.add_argument('--drop_connect_rate', type=float, default=0.0,
    #                     help='connection dropout rate')
    parser.add_argument('--save_config', action='store_true', default=True,
                        help='save config file')


    # 添加参数
    # parser.add_argument('--bmname', dest='bmname',
    #                     help='Name of the benchmark dataset')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
                        help='Ratio of number of graphs testing set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=2,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')  # batch 归一化
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--data_PreMerge', dest='data_PreMerge', action='store_const',
                        const=False, default=True,
                        help='Whether to use submission')
    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')

    parser.add_argument('--pool_sizes', type=str,
                        help='pool_sizes', default='10')
    parser.add_argument('--num_pool_matrix', type=int,
                        help='num_pooling_matrix', default=1)
    parser.add_argument('--min_nodes', type=int,
                        help='min_nodes', default=12)

    parser.add_argument('--weight_decay', type=float,
                        help='weight_decay', default=0.0)
    parser.add_argument('--num_pool_final_matrix', type=int,
                        help='number of final pool matrix', default=0)

    parser.add_argument('--pred_hidden', type=str,
                        help='pred_hidden', default='50')  # 多层 '120_50'

    parser.add_argument('--out_dir', type=str,
                        help='out_dir', default='../experiment')
    parser.add_argument('--num_shuffle', type=int,
                        help='total num_shuffle', default=10)
    parser.add_argument('--shuffle', type=int,
                        help='which shuffle, choose from 0 to 9', default=0)
    parser.add_argument('--concat', type=int,
                        help='whether concat', default=1)
    parser.add_argument('--feat', type=str,
                        help='which feat to use', default='node-label')
    parser.add_argument('--mask', type=int,
                        help='mask or not', default=1)
    parser.add_argument('--norm', type=str,
                        help='Norm for eigens', default='l2')

    parser.add_argument('--with_test', type=int,
                        help='with test or not', default=0)
    parser.add_argument('--con_final', type=int,
                        help='con_final', default=1)
    parser.add_argument('--seed', type=int,
                        help='random seed', default=1)
    parser.add_argument('--randGen', type=bool,
                        help='random generate graph size', default=True)
    # parser.add_argument('--device', type=str,
    #                     help='cpu or cuda', default='cpu')

    # 关于 图塌缩操作的 重要参数
    parser.add_argument('--ModelPara_dir', type=str,
                        help='load model para dir', default='')  # 二次训练模型参数 的位置

    # 这句话可以打印到 log 日志
    parser.add_argument('--add_log', type=str,
                        help='add to log file', default='')  # 二次训练模型参数 的位置

    # 关于 图塌缩操作的 重要参数
    parser.add_argument('--normalize', type=int,
                        help='normalized laplacian or not', default=1)  # 图塌缩时 影响得到的池化矩阵

    # 使用 Car_Hacking_Challenge_Dataset_rev20Mar2021 数据集
    # 生成数据集需要修改的参数
    parser.add_argument('--ds', type=list,
                        help='dynamic or static', default=['D'])  # D or S 车辆动态报文 或者 车辆静止报文
    parser.add_argument('--csv_num', nargs='+', type=int,
                        help='csv num', default=[1, 2])  # 0 or 1 or 2  # csv文件标号
    parser.add_argument('--gs', type=int,
                        help='graph size', default=200)

    parser.add_argument('--regen_dataset', type=bool,
                        help='ReGenerate dataset', default=False)  # 如果图数据集有 就不重新生成数据集 只生成processed数据
    parser.add_argument('--dataset_name', type=str,
                        help='dynamic or static', default='Car_Hacking_Challenge_Dataset_rev20Mar2021')  # 0 or 1 or 2

    parser.add_argument('--msg_smallest_num', type=int,
                        help='the smallest num of msg of a graph', default=200)  # 强化学习 每个步骤取一个图 构成这个图报文最小的条数
    parser.add_argument('--msg_biggest_num', type=int,
                        help='the biggest num of msg of a graph', default=500)  # 强化学习 每个步骤取一个图 构成这个图报文最大的条数

    parser.add_argument('--Di_graph', type=int,
                        help='Whether is Di-graph', default=1)  # 是否是有向图 默认为有向图

    parser.set_defaults(max_nodes=81,
                        feature_type='default',
                        datadir='../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/',
                        lr=0.001,
                        clip=2.0,
                        batch_size=64,
                        num_epochs=100,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=0,
                        # input_dim=10,
                        hidden_dim=20,  # hidden dim
                        output_dim=20,  # embedding dim
                        num_classes=2,  # 分两类 正常报文 异常报文
                        num_gc_layers=3,
                        dropout=0.0,
                        # bmname='Pre_train',
                        )

    cfgs = parser.parse_args()

    cfgs.teacher_model = None

    main(cfgs)

