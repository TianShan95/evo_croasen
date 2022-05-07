#!/usr/bin/env python
# coding=utf-8
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import time
import json
import math
from tqdm import tqdm

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
from torch.autograd import Variable
import sklearn.metrics as metrics
# from imagenet_codebase.utils import *
from ..utils import get_net_info, cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
from myofa.utils import  AverageMeter, accuracy


class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 mixup_alpha,
                 model_init, validation_frequency, print_frequency):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.mixup_alpha = mixup_alpha

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch  # 所有epoch执行的所有batch次数
            T_cur = epoch * nBatch + batch  # 此时是第几个 batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        elif self.lr_schedule_type is None:
            lr = self.init_lr
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self.calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def sample_trainx0(self):
        raise self.data_provider.sample_trainx0

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        return self.data_provider.build_sub_train_loader(n_images, batch_size, num_worker, num_replicas, rank)

    """ optimizer """

    def build_optimizer(self, net_params):
        if self.no_decay_keys is not None:  # None
            assert isinstance(net_params, list) and len(net_params) == 2
            net_params = [
                {'params': net_params[0], 'weight_decay': self.weight_decay},
                {'params': net_params[1], 'weight_decay': 0},
            ]
        else:
            net_params = [{'params': net_params, 'weight_decay': self.weight_decay}]

        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov)
        elif self.opt_type == 'adam':
            optimizer = torch.optim.Adam(net_params, self.init_lr)
        else:
            raise NotImplementedError
        return optimizer


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, args=None, init=True, measure_latency=None, no_gpu=False, mix_prec=None):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.mix_prec = mix_prec

        self.best_acc = 0
        self.best_acc_loss = 0  # 验证集最好的 acc 时的 loss
        self.start_epoch = 0
        self.args = args

        os.makedirs(self.path, exist_ok=True)

        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device('cuda:0')
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        # initialize model (default)
        if init:
            self.network.init_model(run_config.model_init)

        # net info
        # net_info = get_net_info(self.net, self.run_config.data_provider.data_shape, measure_latency, True)
        # with open('%s/net_info.txt' % self.path, 'w') as fout:
        #     fout.write(json.dumps(net_info, indent=4) + '\n')
        #     try:
        #         fout.write(self.network.module_str)
        #     except Exception:
        #         pass

        # criterion
        # if isinstance(self.run_config.mixup_alpha, float):
        #     self.train_criterion = cross_entropy_loss_with_soft_target
        # elif self.run_config.label_smoothing > 0:
        #     self.train_criterion = lambda pred, target: \
        #         cross_entropy_with_label_smoothing(pred, target, self.run_config.label_smoothing)
        # else:
        self.train_criterion = nn.CrossEntropyLoss(size_average=True)
        self.test_criterion = nn.CrossEntropyLoss(size_average=True)

        # optimizer
        if self.run_config.no_decay_keys:  # None
            keys = self.run_config.no_decay_keys.split('#')
            net_params = [
                self.network.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.network.get_parameters(keys, mode='include'),  # parameters without weight decay
            ]
        else:
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = self.network.parameters()
        self.optimizer = self.run_config.build_optimizer(net_params)

        # if mix_prec is not None:
        #     from apex import amp
        #     self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level=mix_prec)

        # self.net = torch.nn.DataParallel(self.net)

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get('_save_path', None) is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self.__dict__['_save_path'] = save_path
        return self.__dict__['_save_path']

    @property
    def logs_path(self):
        if self.__dict__.get('_logs_path', None) is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__['_logs_path'] = logs_path
        return self.__dict__['_logs_path']

    @property
    def network(self):
        if isinstance(self.net, nn.DataParallel):
            return self.net.module
        else:
            return self.net

    @network.setter
    def network(self, new_val):
        if isinstance(self.net, nn.DataParallel):
            self.net.module = new_val
        else:
            self.net = new_val

    def write_log(self, log_str, prefix='valid', should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        elif prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        else:
            with open(os.path.join(self.logs_path, '%s.txt' % prefix), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.network.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        # if self.mix_prec is not None:
        #     from apex import amp
        #     checkpoint['amp'] = amp.state_dict()

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'model': checkpoint['model'], 'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.network.load_state_dict(checkpoint['state_dict'])

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # if self.mix_prec is not None and 'amp' in checkpoint:
            #     from apex import amp
            #     amp.load_state_dict(checkpoint['amp'])

            print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self):
        """ dump run_config and net_config to the model_folder """
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.network.config, open(net_save_path, 'w'), indent=4)
        print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def validate(self, epoch=0, is_test=True, run_str='', net=None, data_loader=None, no_logs=False):
        if net is None:
            net = self.net
        # if not isinstance(net, nn.DataParallel):
        #     net = nn.DataParallel(net)

        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.valid_loader

        net.eval()

        losses = AverageMeter()

        # with torch.no_grad():
        #     with tqdm(total=len(data_loader),
        #               desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
        #         for i, (images, labels) in enumerate(data_loader):
        #             images, labels = images.to(self.device), labels.to(self.device)
                    # compute output
                    # output = net(images)
                    # loss = self.test_criterion(output, labels)
                    # # measure accuracy and record loss
                    # acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        # 移植程序
        labels = []
        preds = []
        acc = 0
        with torch.no_grad():
            with tqdm(total=len(data_loader),
                      desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
                for batch_idx, data in enumerate(data_loader):
                    adj = Variable(data['adj'].float(), requires_grad=False).to(self.device)
                    h0 = Variable(data['feats'].float()).to(self.device)
                    label_cpu = Variable(data['label'].long())
                    label = label_cpu.to(self.device)
                    labels.append(label_cpu)
                    batch_num_nodes = data['num_nodes'].int().numpy()

                    adj_pooled_list = []
                    batch_num_nodes_list = []
                    pool_matrices_dic = dict()
                    pool_sizes = [int(i) for i in self.args.pool_sizes.split('_')]
                    for i in range(len(pool_sizes)):
                        ind = i + 1
                        adj_key = 'adj_pool_' + str(ind)
                        adj_pooled_list.append(Variable(data[adj_key].float(), requires_grad=False).to(self.device))
                        num_nodes_key = 'num_nodes_' + str(ind)
                        batch_num_nodes_list.append(data[num_nodes_key])

                        pool_matrices_list = []
                        for j in range(self.args.num_pool_matrix):
                            pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)

                            pool_matrices_list.append(
                                Variable(data[pool_adj_key].float(), requires_grad=False).to(self.device))

                        pool_matrices_dic[i] = pool_matrices_list

                    pool_matrices_list = []
                    if self.args.num_pool_final_matrix > 0:

                        for j in range(self.args.num_pool_final_matrix):
                            pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                            pool_matrices_list.append(
                                Variable(data[pool_adj_key].float(), requires_grad=False).to(self.device))

                        pool_matrices_dic[ind] = pool_matrices_list

                    ypred, _ = net(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)

                    # else:
                    #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
                    _, indices = torch.max(ypred, 1)
                    preds.append(indices.cpu().data.numpy())

                    loss = self.test_criterion(ypred, label)
                    # measure accuracy and record loss
                    # acc1, acc5 = accuracy(ypred, labels, topk=(1, 5))

                    losses.update(loss.item(), 1)
                    # top1.update(acc1[0].item(), adj.size(0))
                    # top5.update(acc5[0].item(), adj.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'acc': acc,
                    })
                    t.update(1)

            labels = np.hstack(labels)
            preds = np.hstack(preds)

            acc = metrics.accuracy_score(labels, preds)
            print('acc: ', acc)

        return losses.avg, acc

    def train_one_epoch(self, args, epoch, mask_nodes=True):
        # switch to train mode
        self.net.train()

        nBatch = len(self.run_config.train_loader)

        data_time = AverageMeter()
        graph_losses = AverageMeter()
        # cosin
        # if epoch < warmup_epochs:
        #     new_lr = self.run_config.warmup_adjust_learning_rate(
        #         self.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
        #     )
        # else:
        #     new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch - warmup_epochs, i, nBatch)

        # 移植程序
        with tqdm(total=nBatch, desc='Train Epoch #{}'.format(epoch + 1)) as t:
            end = time.time()
            labels = []
            preds = []
            acc = 0
        # with torch.autograd.set_detect_anomaly(True):
            for batch_idx, data in enumerate(self.run_config.train_loader):

                data_time.update(time.time() - end)
                # self.net.zero_grad()
                adj = Variable(data['adj'].float(), requires_grad=False).to(self.device)
                h0 = Variable(data['feats'].float(), requires_grad=False).to(self.device)
                label_cpu = Variable(data['label'].long())
                label = label_cpu.to(self.device)
                labels.append(label_cpu)
                batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None  # 每个原始图的 节点个数
                # assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)
                # if args.method == 'wave':
                adj_pooled_list = []  # 坍缩后的 邻接矩阵 大小为 self.max_node_num
                batch_num_nodes_list = []  # 坍缩后的节点个数
                pool_matrices_dic = dict()
                pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
                for i in range(len(pool_sizes)):
                    ind = i + 1
                    adj_key = 'adj_pool_' + str(ind)  # 坍缩后的 邻接矩阵 大小为 self.max_node_num
                    adj_pooled_list.append(Variable(data[adj_key].float(), requires_grad=False).to(self.device))
                    num_nodes_key = 'num_nodes_' + str(ind)  # 坍缩后的 节点个数
                    batch_num_nodes_list.append(data[num_nodes_key])

                    pool_matrices_list = []
                    for j in range(args.num_pool_matrix):
                        pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)  # 池化矩阵  大小为 self.max_node_num

                        pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(self.device))

                    pool_matrices_dic[i] = pool_matrices_list

                pool_matrices_list = []
                if args.num_pool_final_matrix > 0:  # 默认为 0

                    for j in range(args.num_pool_final_matrix):
                        pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                        pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(self.device))

                    pool_matrices_dic[ind] = pool_matrices_list

                ypred, _ = self.net(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
                _, indices = torch.max(ypred, 1)
                preds.append(indices.cpu().data.numpy())
                # else:
                #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
                # if not args.method == 'soft-assign' or not args.linkpred:
                loss = self.train_criterion(ypred, label)
                # loss = F.cross_entropy(ypred, label, size_average=True)
                # else:
                #     loss = model.loss(ypred, label, adj, batch_num_nodes)
                loss.backward()

                nn.utils.clip_grad_norm(self.net.parameters(), args.clip)
                self.optimizer.step()
                end = time.time()

                # top1, top5 = accuracy(ypred, label, topk=(1, 5))
                graph_losses.update(loss.item(), 1)

                t.set_postfix({
                    'loss': graph_losses.avg,
                    # 'acc': acc,
                    'data_time': data_time.avg,
                })
                t.update(1)

            labels = np.hstack(labels)
            preds = np.hstack(preds)
            acc = metrics.accuracy_score(labels, preds)
            print('acc: ', acc)

        return graph_losses.avg, acc

    def train(self, args):
        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            train_loss, acc = self.train_one_epoch(args, epoch)
            train_log = 'Train [{0}/{1}]\tloss {2:.3f}\tacc {3:.3f} ({4:.3f})'. \
                format(epoch + 1, self.run_config.n_epochs, train_loss, acc, self.best_acc)
            self.write_log(train_log, prefix='train', should_print=False)
            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc = self.validate(net=self.net, epoch=epoch, is_test=False)

                is_best = val_acc > self.best_acc
                self.best_acc = max(self.best_acc, val_acc)
                if is_best:
                    self.best_acc_loss = val_loss
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\tacc {3:.3f} ({4:.3f})'. \
                    format(epoch + 1 , self.run_config.n_epochs, val_loss, val_acc, self.best_acc)
                # val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1:.3f}\tloss {train_loss:.3f}\t'. \
                #     format(np.mean(val_acc5), top1=train_top1, train_loss=train_loss)
                # for v_a in val_acc:
                #     val_log += '(%.3f), ' % v_a
                self.write_log(val_log, prefix='valid', should_print=False)
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.network.state_dict(),
                'model': self.network
            }, is_best=is_best)
            # break

        return self.best_acc_loss, self.best_acc

    # def reset_running_statistics(self, net=None):
    #     from myofa.elastic_nn.utils import set_running_statistics
    #     if net is None:
    #         net = self.network
    #     sub_train_loader = self.run_config.random_sub_train_loader(2000, 100)
    #     set_running_statistics(net, sub_train_loader)
