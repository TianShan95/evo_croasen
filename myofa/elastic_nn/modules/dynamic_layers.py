# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from collections import OrderedDict
import copy

from myofa.layers import MBInvertedConvLayer, ConvLayer, LinearLayer
from myofa.can_codebase.utils import MyModule, MygcnModule, int2list, get_net_device, build_activation
from myofa.elastic_nn.modules.dynamic_op import *
from myofa.elastic_nn.utils import adjust_bn_according_to_idx, copy_bn


class DynamicMBConvLayer(MyModule):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=6, stride=1, act_func='relu6', use_se=False):
        super(DynamicMBConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = int2list(kernel_size_list, 1)
        self.expand_ratio_list = int2list(expand_ratio_list, 1)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se

        # build modules
        max_middle_channel = round(max(self.in_channel_list) * max(self.expand_ratio_list))
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', DynamicPointConv2d(max(self.in_channel_list), max_middle_channel)),
                ('bn', DynamicBatchNorm2d(max_middle_channel)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicSeparableConv2d(max_middle_channel, self.kernel_size_list, self.stride)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', DynamicPointConv2d(max_middle_channel, max(self.out_channel_list))),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = \
                make_divisible(round(in_channel * self.active_expand_ratio), 8)

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.use_se:
            return 'SE(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)
        else:
            return '(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)

    @property
    def config(self):
        return {
            'name': DynamicMBConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size_list': self.kernel_size_list,
            'expand_ratio_list': self.expand_ratio_list,
            'stride': self.stride,
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMBConvLayer(**config)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        middle_channel = make_divisible(round(in_channel * self.active_expand_ratio), 8)

        # build the new layer
        sub_layer = MBInvertedConvLayer(
            in_channel, self.active_out_channel, self.active_kernel_size, self.stride, self.active_expand_ratio,
            act_func=self.act_func, mid_channels=middle_channel, use_se=self.use_se,
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.conv.weight.data[:middle_channel, :in_channel, :, :]
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.fc.reduce.weight.data[:se_mid, :middle_channel, :, :]
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(self.depth_conv.se.fc.reduce.bias.data[:se_mid])

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.fc.expand.weight.data[:middle_channel, :se_mid, :, :]
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(self.depth_conv.se.fc.expand.bias.data[:middle_channel])

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.conv.weight.data[:self.active_out_channel, :middle_channel, :, :]
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = torch.sum(torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3))
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width = sorted_expand_list[expand_ratio_stage]
            target_width = round(max(self.in_channel_list) * target_width)
            importance[target_width:] = torch.arange(0, target_width - importance.size(0), -1)

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )

        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.use_se:
            # se expand: output dim 0 reorganize
            se_expand = self.depth_conv.se.fc.expand
            se_expand.weight.data = torch.index_select(se_expand.weight.data, 0, sorted_idx)
            se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
            # se reduce: input dim 1 reorganize
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 1, sorted_idx)
            # middle weight reorganize
            se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
            se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

            se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
            se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)

        # TODO if inverted_bottleneck is None, the previous layer should be reorganized accordingly
        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            return sorted_idx


class DynamicConvLayer(MyModule):

    def __init__(self, in_channel_list, out_channel_list, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu6'):
        super(DynamicConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list), max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act = build_activation(self.act_func, inplace=True)

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    @property
    def module_str(self):
        return 'DyConv(O%d, K%d, S%d)' % (self.active_out_channel, self.kernel_size, self.stride)

    @property
    def config(self):
        return {
            'name': DynamicConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = ConvLayer(
            in_channel, self.active_out_channel, self.kernel_size, self.stride, self.dilation,
            use_bn=self.use_bn, act_func=self.act_func
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(self.conv.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer


class DynamicLinearLayer(MyModule):

    def __init__(self, in_features_dim, out_features_dim, bias=True, act_func='relu', dropout_rate=0):
        super(DynamicLinearLayer, self).__init__()

        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.bias = bias
        self.dropout_rate = dropout_rate


        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=self.in_features_dim, max_out_features=self.out_features_dim, bias=self.bias
        )

        self.act = build_activation(act_func, inplace=False)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.linear(x))

    @property
    def module_str(self):
        return 'DyLinear(%d)' % self.out_features_dim

    @property
    def config(self):
        return {
            'name': DynamicLinear.__name__,
            'in_features_dim': self.in_features_dim,
            'out_features': self.out_features_dim,
            'bias': self.bias
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_dim, out_dim, act_func='relu', dropout_rate=0, preserve_weight=True):
        # 预测层 不同的层神经元数可以乘以不同的 比例
        sub_layer = DynamicLinearLayer(in_dim, out_dim, self.bias, act_func, dropout_rate=dropout_rate)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.linear.weight.data.copy_(self.linear.linear.weight.data[:out_dim, :in_dim])
        if self.bias:
            sub_layer.linear.linear.bias.data.copy_(self.linear.linear.bias.data[:out_dim])
        return sub_layer


class DynamicGcnLayer(MygcnModule):

    def __init__(self, input_dim, output_dim, act_func, device, add_self=False, normalize_embedding=False,
                 dropout_rate=0.0, with_bias=True, bn=True):
        super(DynamicGcnLayer, self).__init__()

        self.add_self = add_self
        self.dropout_rate = dropout_rate
        self.act_func = act_func
        self.with_bias = with_bias
        self.bn = bn
        self.device = device

        if self.dropout_rate > 0:
            self.dropout_layer = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout_layer = None

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if with_bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None

        self.act = build_activation(self.act_func, inplace=True)

        # # 初始化权重值
        # for m in self.modules():
        #     if isinstance(m, MygcnModule):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(self.act_func))
        #         if m.bias is not None:
        #             m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(int(x.size()[1])).to(self.device)
        return bn_module(x)

    def forward(self, x, adj):
        if self.dropout_rate > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x

        y = torch.matmul(y, self.weight)

        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        y = self.act(y)
        if self.bn:
            y = self.apply_bn(y)

        return y

    @property
    def module_str(self):
        return 'DyGcn(%d)' % self.output_dim

    @property
    def config(self):
        return {
            'name': DynamicGcnLayer.__name__,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'bias': self.with_bias
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, act, dropout_rate, preserve_weight=True):

        # build new layer
        sub_layer = DynamicGcnLayer(self.input_dim, self.output_dim, act_func=act, device=self.device, add_self=self.add_self,
                                    normalize_embedding=self.normalize_embedding,
                                    dropout_rate=dropout_rate, with_bias=self.with_bias)

        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.weight.data.copy_(self.weight.data[:, :])
        if self.with_bias:
            sub_layer.bias.data.copy_(self.bias.data[:])
        return sub_layer