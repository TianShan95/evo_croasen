# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

import torch

from myofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer, DynamicGcnLayer
from myofa.layers import ConvLayer, IdentityLayer, LinearLayer, MBInvertedConvLayer
# from ofa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3, MobileInvertedResidualBlock
from myofa.can_codebase.utils import make_divisible, int2list

from myofa.can_codebase.networks.coarsengcn_v1 import CoarsenGcnV1


class OFACoarsenGcnV1(CoarsenGcnV1):

    def __init__(self, n_classes=2, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None,
                 width_mult_list=1.0, depth_list=4, out_gcn_vector=None, args=None):


        self.depth_list = int2list(depth_list, 1)
        self.width_mult_list = int2list(width_mult_list, 1)
        # self.dropout_rate = dropout_rate
        self.out_gcn_vector = out_gcn_vector
        self.n_classes = n_classes
        self.args = args

        # device
        self.device = 'cpu'
        # move network to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

        n_block_list = [max(self.depth_list)] * 2  # [3, 3]


        # inverted residual blocks
        self.block_group_info = []
        blocks = torch.nn.ModuleList()  # 存储 块
        # layers = torch.nn.ModuleList()  # 存储 层
        _block_index = 0

        gcn_hidden_dim = [20, 20, 20, 20, 20, 20]
        gcn_input_dim = [81, (max(gcn_hidden_dim))]
        self.base_linear_hidden_dim = [360, 720]
        self.runtime_linear_hidden_dim = [360, 720]
        self.runtime_act_stages = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']

        # 图卷积层
        for n_block_index, n_block in enumerate(n_block_list):  # 遍历图卷积块
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block
            input_dim = gcn_input_dim[n_block_index]
            for i in range(n_block):  # 遍历每个图卷积block里面的层
                output_dim = gcn_hidden_dim[n_block_index*max(self.depth_list)+i]
                gcn_conv = DynamicGcnLayer(input_dim=input_dim, output_dim=output_dim, act_func=self.runtime_act_stages[i], device=self.device)  # 建立 图卷积层
                # layers.append(gcn_conv)
                input_dim = output_dim
                blocks.append(gcn_conv)
            # blocks.append(copy.deepcopy(layers))
            # layers = torch.nn.ModuleList()

        # 预测层
        # linear_blocks = torch.nn.ModuleList()
        linear_blocks = torch.nn.Sequential()
        self.linear_hidden_dim = []  # 全连接层 经过 width_ratio 的维度
        self.act_stages = []  # 每个子网络所使用的 激活函数
        self.linear_init_input_dim = 40  # 预测层初始输入为 40
        linear_input_dim = self.linear_init_input_dim
        for h_dim_index, h_dim in enumerate(self.runtime_linear_hidden_dim):
            linear_output_dim = int(h_dim*self.width_mult_list[h_dim_index])
            act = self.runtime_act_stages[len(n_block_list)*max(self.depth_list)+h_dim_index]

            # self.run_time_linear_hidden_dim.append(linear_output_dim)
            # self.run_time_act_stages.append(act)

            classifier = DynamicLinearLayer(
                in_features_dim=linear_input_dim, out_features_dim=linear_output_dim, bias=True,
                act_func=act, dropout_rate=dropout_rate
            )
            # linear_blocks.append(classifier)
            linear_blocks.add_module('linear_%d' % h_dim_index, classifier)
            linear_input_dim = linear_output_dim
        # 最后一层 全连接层
        classifier = DynamicLinearLayer(
            in_features_dim=linear_input_dim, out_features_dim=n_classes, bias=True,
            act_func='softmax', dropout_rate=dropout_rate
        )
        # linear_blocks.append(classifier)
        linear_blocks.add_module('last_linear', classifier)

        super(OFACoarsenGcnV1, self).__init__(blocks, linear_blocks, n_block_list, dropout_rate, out_gcn_vector, args.num_pool_matrix, args.mask, args.con_final, args.num_pool_final_matrix)

        # set bn param
        # self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFACoarsenGcnV1'

    # def forward(self, x):
    #     # first conv
    #     x = self.first_conv(x)
    #     # first block
    #     x = self.blocks[0](x)
    #
    #     # blocks
    #     for stage_id, block_idx in enumerate(self.block_group_info):
    #         depth = self.runtime_depth[stage_id]
    #         active_idx = block_idx[:depth]
    #         for idx in active_idx:
    #             x = self.blocks[idx](x)
    #
    #     x = self.final_expand_layer(x)
    #     x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
    #     x = self.feature_mix_layer(x)
    #     x = torch.squeeze(x)
    #     x = self.classifier(x)
    #     return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.gcn_blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.gcn_blocks[idx].module_str + '\n'

        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier_blocks.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': OFACoarsenGcnV1.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.gcn_blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier_blocks.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_weights_from_net(self, src_model_dict):
        model_dict = self.state_dict()
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = src_model_dict[key]
        self.load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, depth, width_rate, activation, dropout_rate, out_gcn_vector):

        # depth = int2list(d, len(self.block_group_info))
        # 确定每个卷积块需要的卷积层数
        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        # width_rate = int2list(wr, 1)
        # 确定每个全连接层的神经元数
        for i, d in enumerate(width_rate):
            if d is not None:
                self.runtime_linear_hidden_dim[i] = d*self.base_linear_hidden_dim[i]

        for i, d in enumerate(out_gcn_vector):
            if d is not None:
                self.out_gcn_vector[i] = d

        # act_list = int2list(act, 1)
        # 确定每个每一层的激活函数类型
        for i, d in enumerate(activation):
            if d is not None:
                self.runtime_act_stages[i] = d

        self.dropout_rate = dropout_rate

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'depth':
            self.__dict__['_depth_include_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        elif constraint_type == 'width_mult':
            self.__dict__['_widthMult_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_ks_include_list'] = None
        self.__dict__['_widthMult_include_list'] = None

    def sample_active_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample width_mult
        width_mult_setting = None

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.gcn_blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.gcn_blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):

        gcn_blocks = torch.nn.ModuleList()
        # linear_blocks = torch.nn.ModuleList()
        linear_blocks = torch.nn.Sequential()
        # classifier = copy.deepcopy(self.classifier)

        # gcn_blocks
        for stage_id, block_idx in enumerate(self.block_group_info):  # self.block_group_info = [[0, 1, 2], [3, 4, 5]]
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]  # 添加激活的层
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(self.gcn_blocks[idx].get_active_subnet(self.runtime_act_stages[idx], self.dropout_rate, preserve_weight))  # 激活子网络
            gcn_blocks += stage_blocks

        # 全连接层
        linear_input_dim = self.linear_init_input_dim
        for h_dim_index, h_dim in enumerate(self.runtime_linear_hidden_dim):
            linear_output_dim = int(h_dim*self.width_mult_list[h_dim_index])
            act = self.runtime_act_stages[len(self.block_group_info)*max(self.depth_list)+h_dim_index]

            # self.run_time_linear_hidden_dim.append(linear_output_dim)
            # self.run_time_act_stages.append(act)
            linear_blocks.add_module('linear_%d' % h_dim_index, self.classifier_blocks[h_dim_index].get_active_subnet(linear_input_dim, h_dim, act, self.dropout_rate, preserve_weight=True))

            # classifier = DynamicLinearLayer(
            #     in_features_dim=linear_input_dim, out_features=linear_output_dim, bias=True,
            #     act_func=act, dropout_rate=self.dropout_rate
            # )
            #
            #
            # linear_blocks.append(classifier)
            linear_input_dim = linear_output_dim
        # 最后一层 全连接层
        # classifier = DynamicLinearLayer(
        #     in_features_dim=linear_input_dim, out_features=self.n_classes, bias=True,
        #     act_func='softmax', dropout_rate=self.dropout_rate
        # )
        linear_blocks.add_module('last_linear', self.classifier_blocks[-1].get_active_subnet(linear_input_dim, self.n_classes, act_func='softmax', dropout_rate=self.dropout_rate, preserve_weight=True))

        # import hiddenlayer as hl
        # # 构建一个 gcn 模型
        # # gcn_blocks[0].eval()
        # linear_net = torch.nn.Sequential()
        # for linear_id, linear_layer in enumerate(linear_blocks):
        #     print(linear_layer.linear.linear)
        #     print(linear_layer.dropout)
        #     linear_net.add_module("drop_%d" % linear_id, linear_layer.dropout)
        #     linear_net.add_module("linear_%d" % linear_id, linear_layer.linear)
        #     linear_net.add_module("act_%d" % linear_id, linear_layer.act)
        #
        # x_trace = torch.zeros([64, 40]).float().to(self.device)
        # # adj_trace = torch.zeros([64, 81, 81]).to(self.device)
        # gcn_graph = hl.build_graph(linear_net, x_trace)
        # gcn_graph.theme = hl.graph.THEMES["blue"].copy()
        # gcn_graph.save(f"linear_blocks.png", format='png')


        _subnet = CoarsenGcnV1(gcn_blocks, linear_blocks, self.runtime_depth, self.dropout_rate, self.out_gcn_vector, self.num_pool_matrix, self.args.mask, self.args.con_final, self.args.num_pool_final_matrix)
        # _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        # first conv
        first_conv_config = self.first_conv.config
        first_block_config = self.gcn_blocks[0].config

        final_expand_config = self.final_expand_layer.config
        feature_mix_layer_config = self.feature_mix_layer.config
        if isinstance(self.final_expand_layer, DynamicConvLayer):
            final_expand_config = self.final_expand_layer.get_active_subnet_config(
                self.gcn_blocks[-1].mobile_inverted_conv.active_out_channel)
            feature_mix_layer_config = self.feature_mix_layer.get_active_subnet_config(
                final_expand_config['out_channels'])
        classifier_config = self.classifier_blocks.config
        if isinstance(self.classifier_blocks, DynamicLinearLayer):
            classifier_config = self.classifier_blocks.get_active_subnet_config(self.feature_mix_layer.active_out_channel)

        block_config_list = [first_block_config]
        input_channel = first_block_config['mobile_inverted_conv']['out_channels']
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                middle_channel = make_divisible(round(input_channel *
                                                      self.gcn_blocks[idx].mobile_inverted_conv.active_expand_ratio), 8)
                stage_blocks.append({
                    # 'name': MobileInvertedResidualBlock.__name__,
                    'mobile_inverted_conv': {
                        'name': MBInvertedConvLayer.__name__,
                        'in_channels': input_channel,
                        'out_channels': self.gcn_blocks[idx].mobile_inverted_conv.active_out_channel,
                        'kernel_size': self.gcn_blocks[idx].mobile_inverted_conv.active_kernel_size,
                        'stride': self.gcn_blocks[idx].mobile_inverted_conv.stride,
                        'expand_ratio': self.gcn_blocks[idx].mobile_inverted_conv.active_expand_ratio,
                        'mid_channels': middle_channel,
                        'act_func': self.gcn_blocks[idx].mobile_inverted_conv.act_func,
                        'use_se': self.gcn_blocks[idx].mobile_inverted_conv.use_se,
                    },
                    'shortcut': self.gcn_blocks[idx].shortcut.config if self.gcn_blocks[idx].shortcut is not None else None,
                })
                input_channel = self.gcn_blocks[idx].mobile_inverted_conv.active_out_channel
            block_config_list += stage_blocks

        return {
            'name': CoarsenGcnV1.__name__,
            'bn': self.get_bn_param(),
            'first_conv': first_conv_config,
            'blocks': block_config_list,
            'final_expand_layer': final_expand_config,
            'feature_mix_layer': feature_mix_layer_config,
            'classifier': classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.gcn_blocks[1:]:
            block.mobile_inverted_conv.re_organize_middle_weights(expand_ratio_stage)
