import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from logger.logger import logger



class GcnEncoderGraph(nn.Module):
    def __init__(self, concat=True, bn=True, device='cpu'):
        '''
        :param concat:
        :param bn:
        :param device:
        '''
        super(GcnEncoderGraph, self).__init__()

        logger.info(f'Whether concat {concat}')

        self.device = device

        self.concat = concat
        add_self = not concat
        self.bn = bn
        # self.num_layers = num_layers
        self.num_aggs = 1

        # self.bias = True
        # if args is not None:
        #     self.bias = args.bias

        # self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
        #     input_dim, hidden_dim, embedding_dim, num_layers,
        #     add_self, normalize=True, dropout=dropout)
        # self.act = nn.ReLU().to(device)
        # self.label_dim = label_dim

        # if concat:
        #     self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        # else:
        #     self.pred_input_dim = embedding_dim
        # self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
        #                                          label_dim, num_aggs=self.num_aggs)

        # for m in self.modules():
        #     if isinstance(m, GraphConv):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         if m.bias is not None:
        #             m.bias.data = nn.init.constant_(m.bias.data, 0.0)

        # logger.info(f'num_layers: {num_layers}')
        # logger.info(f'pred_hidden_dims: {pred_hidden_dims}')
        # logger.info(f'hidden_dim: {hidden_dim}')
        # logger.info(f'embedding_dim: {embedding_dim}')
        # logger.info(f'label_dim {label_dim}')

    # def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
    #                       normalize=False, dropout=0.0):
    #     conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
    #                            normalize_embedding=normalize, bias=self.bias, device=self.device)
    #     conv_block = nn.ModuleList(
    #         [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
    #                    normalize_embedding=normalize, dropout=dropout, bias=self.bias, device=self.device)
    #          for i in range(num_layers - 2)])
    #     conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
    #                           normalize_embedding=normalize, bias=self.bias, device=self.device)
    #     return conv_first, conv_block, conv_last

    # def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
    #
    #     pred_input_dim = pred_input_dim * num_aggs
    #     if len(pred_hidden_dims) == 0:
    #         pred_model = nn.Linear(pred_input_dim, label_dim).to(self.device)
    #     else:
    #         pred_layers = []
    #         for pred_dim in pred_hidden_dims:  # 全连接 中间隐藏层维度
    #             pred_layers.append(nn.Linear(pred_input_dim, pred_dim).to(self.device))
    #             pred_layers.append(self.act)
    #             pred_input_dim = pred_dim
    #         pred_layers.append(nn.Linear(pred_dim, label_dim).to(self.device))
    #         pred_model = nn.Sequential(*pred_layers)
    #     return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)).to(self.device) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes).to(self.device)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)

    def gcn_forward(self, x, adj, gcn_blocks, out_gcn_vector='sum', embedding_mask=None):
    # 把gcnblocks传入进来 进行图卷积
        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x_all = []
        for gcn_net in gcn_blocks:
            x = gcn_net(x, adj)
            if self.bn:
                x = self.apply_bn(x)  # batch norm
            x_all.append(x)

        # x = conv_first(x, adj)  # 第一层卷积
        # x = self.act(x)  # 激活函数

        # if self.bn:
        #     x = self.apply_bn(x)  # batch norm

        # x_all = [x]

        # for i in range(len(conv_block)):  # 中间层卷积
        #     x = conv_block[i](x, adj)
        #     x = self.act(x)
        #     if self.bn:
        #         x = self.apply_bn(x)
        #     x_all.append(x)

        # if conv_last:  # 可能存在最后一层卷积被屏蔽的情况
        #     x = conv_last(x, adj)  # 最后 一层卷积
        #     x_all.append(x)

        # if self.concat:
        #     x_tensor = torch.cat(x_all, dim=2)
        # else:
        #     x_tensor = x
        #

        x_tensor= None
        if out_gcn_vector == 'sum':
            x_tensor = torch.sum(torch.tensor(x_all), dim=2)
        elif out_gcn_vector == 'mean':
            x_tensor = torch.mean(torch.tensor(x_all), dim=2)
        elif out_gcn_vector == 'max':
            x_tensor = torch.max(torch.tensor(x_all), dim=2)

        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask

        return x_tensor

    # def forward(self, x, adj, batch_num_nodes=None, **kwargs):
    #
    #     max_num_nodes = adj.size()[1]
    #
    #     if batch_num_nodes is not None:
    #         self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
    #     else:
    #         self.embedding_mask = None
    #
    #     x = self.conv_first(x, adj)
    #
    #     x = self.act(x)
    #
    #     if self.bn:
    #         x = self.apply_bn(x)
    #
    #     out_all = []
    #     out, _ = torch.max(x, dim=1)
    #
    #     out_all.append(out)
    #     for i in range(self.num_layers - 2):
    #         x = self.conv_block[i](x, adj)
    #         x = self.act(x)
    #         if self.bn:
    #             x = self.apply_bn(x)
    #         out, _ = torch.max(x, dim=1)
    #         out_all.append(out)
    #         if self.num_aggs == 2:
    #             out = torch.sum(x, dim=1)
    #             out_all.append(out)
    #     x = self.conv_last(x, adj)
    #
    #     out, _ = torch.max(x, dim=1)
    #     out_all.append(out)
    #     if self.num_aggs == 2:
    #         out = torch.sum(x, dim=1)
    #         out_all.append(out)
    #     if self.concat:
    #         output = torch.cat(out_all, dim=1)
    #     else:
    #         output = out
    #
    #     ypred = self.pred_model(output)
    #
    #     return ypred

    # def loss(self, pred, label, type='softmax'):
    #
    #     if type == 'softmax':
    #         return F.cross_entropy(pred, label, size_average=True)
    #     elif type == 'margin':
    #         batch_size = pred.size()[0]
    #         label_onehot = torch.zeros(batch_size, self.label_dim).long().to(self.device)
    #         label_onehot.scatter_(1, label.view(-1, 1), 1)
    #         return torch.nn.MultiLabelMarginLoss()(pred, label_onehot).to(self.device)


class WavePoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, num_pool_matrix=1, args=None, device='cpu'):

        super(WavePoolingGcnEncoder, self).__init__()
        # add_self = not concat
        # self.mask = mask
        # self.pool_sizes = [pool_sizes]
        # self.num_pool_matrix = num_pool_matrix  # args 默认为 1
        # self.num_pool_final_matrix = num_pool_final_matrix

        # self.con_final = args.con_final

        # self.device = device

        # logger.info(f'Device_-wave: {device}')

        # self.conv_first_after_pool = nn.ModuleList()
        # self.conv_block_after_pool = nn.ModuleList()
        # self.conv_last_after_pool = nn.ModuleList()
        # for i in range(len(self.pool_sizes)):
        #     logger.info(f'In WavePooling {self.pred_input_dim * self.num_pool_matrix}')
            # conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
            #     self.pred_input_dim * self.num_pool_matrix, hidden_dim, embedding_dim, num_layers,
            #     add_self, normalize=True, dropout=dropout)

            # self.conv_first_after_pool.append(conv_first2)
            # self.conv_block_after_pool.append(conv_block2)
            # self.conv_last_after_pool.append(conv_last2)

        # if self.num_pool_final_matrix > 0:
        #     if concat:
        #
        #         if self.con_final:
        #             self.pred_model = self.build_pred_layers(
        #                 self.pred_input_dim * (len(pool_sizes) + 1) + self.pred_input_dim * self.num_pool_final_matrix,
        #                 pred_hidden_dims,
        #                 label_dim, num_aggs=self.num_aggs)
        #         else:
        #             self.pred_model = self.build_pred_layers(
        #                 self.pred_input_dim * (len(pool_sizes)) + self.pred_input_dim * self.num_pool_final_matrix,
        #                 pred_hidden_dims,
        #                 label_dim, num_aggs=self.num_aggs)
        #
        #
        #     else:
        #
        #         self.pred_model = self.build_pred_layers(self.pred_input_dim * self.num_pool_final_matrix,
        #                                                  pred_hidden_dims,
        #                                                  label_dim, num_aggs=self.num_aggs)
        #
        # else:
        #     if concat:  # 构建预测全连接层
        #         # self.pred_model = self.build_pred_layers(self.pred_input_dim * (len(pool_sizes) + 1), pred_hidden_dims,
        #         #                                          label_dim, num_aggs=self.num_aggs)
        #         self.pred_model = self.build_pred_layers(60, pred_hidden_dims,
        #                                                  label_dim, num_aggs=self.num_aggs)
        #     else:
        #         self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
        #                                                  label_dim, num_aggs=self.num_aggs)
        # for m in self.modules():
        #     if isinstance(m, GraphConv):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         if m.bias is not None:
        #             m.bias.data = nn.init.constant_(m.bias.data, 0.0)






class Pool(nn.Module):
    def __init__(self, num_pool, pool_matrices, device='cpu'):
        super(Pool, self).__init__()

        self.pool_matrices = pool_matrices  # 池化矩阵
        self.num_pool = num_pool  # 池化的次数

        self.device = device

    def forward(self, x):
        pooling_results = [0] * self.num_pool
        for i in range(self.num_pool):
            pool_matrix = self.pool_matrices[i]
            pool_matrix = pool_matrix.type(torch.FloatTensor).to(self.device)

            pool_matrix = torch.transpose(pool_matrix, 1, 2)

            pooling_results[i] = torch.matmul(pool_matrix, x)
        if len(pooling_results) > 1:

            x_pooled = torch.cat([*pooling_results], 2)

        else:
            x_pooled = pooling_results[0]

        return x_pooled
