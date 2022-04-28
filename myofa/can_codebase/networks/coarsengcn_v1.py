# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# from layers import *
from myofa.layers import set_layer_from_config, MBInvertedConvLayer, ConvLayer, IdentityLayer, LinearLayer
# from ofa.imagenet_codebase.utils import MyNetwork, make_divisible
# from ofa.imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock

from myofa.utils import MyNetwork
from myofa.can_codebase.networks.graph_encoders import WavePoolingGcnEncoder
from myofa.can_codebase.networks.graph_encoders import Pool


class CoarsenGcnV1(MyNetwork):

    def __init__(self, gcn_blocks, classifier_layers, gcn_block_depth_list, dropout_rate, out_gcn_vector, num_pool_matrix, mask, con_final, num_pool_final_matrix):
        super(CoarsenGcnV1, self).__init__()

        self.gcn_blocks = gcn_blocks
        self.classifier_blocks = classifier_layers

        self.con_final = con_final
        self.num_pool_final_matrix = num_pool_final_matrix
        self.gcn_block_depth_list = gcn_block_depth_list
        self.pool_sizes = [10]  # 目前只考虑一次图卷积
        self.device = 'cpu'
        self.dropout_rate = dropout_rate
        self.num_pool_matrix = num_pool_matrix  # args 默认为 1
        self.mask = mask
        self.out_gcn_vector = out_gcn_vector
        # move network to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

    def forward(self, x, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic, **kwargs):

        max_num_nodes = adj.size()[1]  # 最大节点数 在 args 处 设置 adj经过了padding处理 大小为 设置的最大节点个数


        # 模型开始处
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)  # batch_num_nodes 原始图的节点个数 embedding_mask batchsize 行 最大节点数列 存在节点的位置置1
        else:
            embedding_mask = None

        out_all = []

        # TODO: out_gcn_vector 把字符串转换为列表
        out_gcn_vector = self.out_gcn_vector
        # 池化之前图卷积图卷积

        embedding_tensor = self.gcn_forward(x, adj, self.gcn_blocks[:self.gcn_block_depth_list[0]], out_gcn_vector[0], embedding_mask, self.device)


        out, _ = torch.max(embedding_tensor, dim=1)  # 维度60 每层卷积输出维度20 三层卷积 concat [batch_size, hidden_dim * gcn_layer_num]

        out_all.append(out)

        for i in range(len(self.pool_sizes)):  # 进行一次图坍缩
            pool = Pool(self.num_pool_matrix, pool_matrices_dic[i], device=self.device)

            embedding_tensor = pool(embedding_tensor)  # 池化后的 x
            if self.mask:  # args 给出 默认为 1
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes_list[i])  # 屏蔽节点 每行代表一个图 每行1的个数 是 坍缩后图节点的个数
            else:
                embedding_mask = None
            adj_new = adj_pooled_list[i].type(torch.FloatTensor).to(self.device)
            # 池化后图卷积
            embedding_tensor = self.gcn_forward(embedding_tensor, adj_new, self.gcn_blocks[self.gcn_block_depth_list[1]:], out_gcn_vector[1], embedding_mask, self.device)

            if self.con_final or self.num_pool_final_matrix == 0:
                out, _ = torch.max(embedding_tensor, dim=1)
                out_all.append(out)

        if self.num_pool_final_matrix > 0:
            pool = Pool(self.num_pool_final_matrix, pool_matrices_dic[i + 1], device=self.device)
            embedding_tensor = pool(embedding_tensor)
            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)

        # if self.concat:
        output = torch.cat(out_all, dim=1)
        # else:
        #     output = out

        # 预测层
        # x = torch.squeeze(output)
        y_pred = output
        for classifier_layer in self.classifier_blocks:
            y_pred = classifier_layer(y_pred)

        # y_pred = self.pred_model(output)
        return y_pred, output

    def loss(self, pred, label):
        return F.cross_entropy(pred, label, size_average=True)

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.gcn_blocks:
            _str += block.module_str + '\n'
        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier_blocks.module_str
        return _str

    @property
    def config(self):
        return {
            'name': CoarsenGcnV1.__name__,
            # 'bn': self.get_bn_param(),
            # 'first_conv': self.first_conv.config,
            'gcn_blocks': [
                block.config for block in self.gcn_blocks
            ],
            # 'final_expand_layer': self.final_expand_layer.config,
            # 'feature_mix_layer': self.feature_mix_layer.config,
            'classifier_blocks': [
                block.config for block in self.classifier_blocks
            ],
        }

    # @staticmethod
    # def build_from_config(config):
    #     first_conv = set_layer_from_config(config['first_conv'])
    #     final_expand_layer = set_layer_from_config(config['final_expand_layer'])
    #     feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
    #     classifier = set_layer_from_config(config['classifier'])
    #
    #     blocks = []
    #     for block_config in config['blocks']:
    #         blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))
    #
    #     net = CoarsenGcnV1(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
    #     if 'bn' in config:
    #         net.set_bn_param(**config['bn'])
    #     else:
    #         net.set_bn_param(momentum=0.1, eps=1e-3)
    #
    #     return net

    # def zero_last_gamma(self):
    #     for m in self.modules():
    #         if isinstance(m, MobileInvertedResidualBlock):
    #             if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
    #                 m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

    # @staticmethod
    # def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
    #     # first conv layer
    #     first_conv = ConvLayer(
    #         3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
    #     )
    #     # build mobile blocks
    #     feature_dim = input_channel
    #     blocks = []
    #     for stage_id, block_config_list in cfg.items():
    #         for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
    #             mb_conv = MBInvertedConvLayer(
    #                 feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se
    #             )
    #             if stride == 1 and out_channel == feature_dim:
    #                 shortcut = IdentityLayer(out_channel, out_channel)
    #             else:
    #                 shortcut = None
    #             blocks.append(MobileInvertedResidualBlock(mb_conv, shortcut))
    #             feature_dim = out_channel
    #     # final expand layer
    #     final_expand_layer = ConvLayer(
    #         feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
    #     )
    #     feature_dim = feature_dim * 6
    #     # feature mix layer
    #     feature_mix_layer = ConvLayer(
    #         feature_dim, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
    #     )
    #     # classifier
    #     classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
    #
    #     return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    # @staticmethod
    # def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
    #     for i, (stage_id, block_config_list) in enumerate(cfg.items()):
    #         for block_config in block_config_list:
    #             if ks is not None and stage_id != '0':
    #                 block_config[0] = ks
    #             if expand_ratio is not None and stage_id != '0':
    #                 block_config[-1] = expand_ratio
    #                 block_config[1] = None
    #                 if stage_width_list is not None:
    #                     block_config[2] = stage_width_list[i]
    #         if depth_param is not None and stage_id != '0':
    #             new_block_config_list = [block_config_list[0]]
    #             new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
    #             cfg[stage_id] = new_block_config_list
    #     return cfg
