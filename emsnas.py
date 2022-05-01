import os
import json
import re
import shutil
import argparse
import subprocess
import numpy as np
from utils import get_correlation
from evaluator import OFAEvaluator, get_net_info

from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.factory import get_performance_indicator
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_algorithm, get_crossover, get_mutation

from search_space.myofa import OFASearchSpace
from acc_predictor.factory import get_acc_predictor
from utils import prepare_eval_folder, MySampling, BinaryCrossover, MyMutation

_DEBUG = False
if _DEBUG: from pymoo.visualization.scatter import Scatter


# EMS NAS
# Evolutionary Multi-Object Surrogate-Assisted
# Neural Architecture Search
class EMsNAS:
    def __init__(self, kwargs):
        self.search_space = OFASearchSpace()
        self.save_path = kwargs.pop('save', '.tmp')  # path to save results
        self.resume = kwargs.pop('resume', None)  # resume search from a checkpoint
        self.sec_obj = kwargs.pop('sec_obj', 'params')  # second objective to optimize simultaneously
        self.iterations = kwargs.pop('iterations', 30)  # number of iterations to run search 搜索的 代数
        self.n_doe = kwargs.pop('n_doe',
                                100)  # number of architectures to train before fit surrogate model fit代理模型之前 要训练架构的数量
        self.n_iter = kwargs.pop('n_iter', 8)  # number of architectures to train in each iteration 每个迭代中要训练架构的数量
        self.predictor = kwargs.pop('predictor', 'rbf')  # which surrogate model to fit
        self.n_gpus = kwargs.pop('n_gpus', 1)  # number of available gpus
        self.gpu = kwargs.pop('gpu', 1)  # required number of gpus per evaluation job
        self.data = kwargs.pop('data', '../data')  # location of the data files
        self.dataset = kwargs.pop('dataset', 'imagenet')  # which dataset to run search on
        self.n_classes = kwargs.pop('n_classes', 2)  # number of classes of the given dataset
        self.n_workers = kwargs.pop('n_workers', 6)  # number of threads for dataloader
        self.vld_size = kwargs.pop('vld_size', 10000)  # number of images from train set to validate performance
        self.trn_batch_size = kwargs.pop('trn_batch_size', 96)  # batch size for SGD training
        self.vld_batch_size = kwargs.pop('vld_batch_size', 250)  # batch size for validation
        self.n_epochs = kwargs.pop('n_epochs', 5)  # number of epochs to SGD training
        self.test = kwargs.pop('test', False)  # evaluate performance on test set
        self.supernet_path = kwargs.pop(
            'supernet_path', '../experiment/evo_croasen/super_net/super_model_best.pth.tar')  # supernet model path
        self.latency = self.sec_obj if "cpu" in self.sec_obj or "gpu" in self.sec_obj else None

    def search(self):

        if self.resume:  # load model from dir
            archive = self._resume_from_dir()
        else:
            # the following lines corresponding to Algo 1 line 1-7 in the paper
            archive = []  # initialize an empty archive to store all trained CNNs

            # Design Of Experiment
            if self.iterations < 1:
                arch_doe = self.search_space.sample1(self.n_doe)  # 采样架构
            else:
                arch_doe = self.search_space.initialize1(self.n_doe)  #

            # parallel evaluation of arch_doe
            top1_err, complexity = self._evaluate(arch_doe, it=0)  # 评估 错误率 复杂度 从训练结果文件得到

            # store evaluated / trained architectures
            for member in zip(arch_doe, top1_err, complexity):
                archive.append(member)

        # reference point (nadir point) for calculating hypervolume
        ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[2] for x in archive])])

        # main loop of the search
        for it in range(1, self.iterations + 1):

            # construct accuracy predictor surrogate model from archive
            # Algo 1 line 9 / Fig. 3(a) in the paper
            acc_predictor, a_top1_err_pred = self._fit_acc_predictor(archive)  # 代理预测精度 top1误差预测

            # search for the next set of candidates for high-fidelity evaluation (lower level)
            # Algo 1 line 10-11 / Fig. 3(b)-(d) in the paper
            candidates, c_top1_err_pred = self._next(archive, acc_predictor, self.n_iter)  # 多目标选择架构 从a～提取子网

            # high-fidelity evaluation (lower level)
            # Algo 1 line 13-14 / Fig. 3(e) in the paper
            c_top1_err, complexity = self._evaluate(candidates, it=it)  # 传入候选架构 和 迭代次数

            # check for accuracy predictor's performance 检查预测器的准确性
            rmse, rho, tau = get_correlation(
                np.vstack((a_top1_err_pred, c_top1_err_pred)), np.array([x[1] for x in archive] + c_top1_err))

            # add to archive
            # Algo 1 line 15 / Fig. 3(e) in the paper
            for member in zip(candidates, c_top1_err, complexity):
                archive.append(member)

            # calculate hypervolume
            hv = self._calc_hv(
                ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive])))

            # print iteration-wise statistics
            print("Iter {}: hv = {:.2f}".format(it, hv))
            print("fitting {}: RMSE = {:.4f}, Spearman's Rho = {:.4f}, Kendall’s Tau = {:.4f}".format(
                self.predictor, rmse, rho, tau))

            # dump the statistics
            with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                json.dump({'archive': archive, 'candidates': archive[-self.n_iter:], 'hv': hv,
                           'surrogate': {
                               'model': self.predictor, 'name': acc_predictor.name,
                               'winner': acc_predictor.winner if self.predictor == 'as' else acc_predictor.name,
                               'rmse': rmse, 'rho': rho, 'tau': tau}}, handle)
            if _DEBUG:
                # plot
                plot = Scatter(legend={'loc': 'lower right'})
                F = np.full((len(archive), 2), np.nan)
                F[:, 0] = np.array([x[2] for x in archive])  # second obj. (complexity)
                F[:, 1] = 100 - np.array([x[1] for x in archive])  # top-1 accuracy
                plot.add(F, s=15, facecolors='none', edgecolors='b', label='archive')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                F[:, 1] = 100 - np.array(c_top1_err)
                plot.add(F, s=30, color='r', label='candidates evaluated')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                F[:, 1] = 100 - c_top1_err_pred[:, 0]
                plot.add(F, s=20, facecolors='none', edgecolors='g', label='candidates predicted')
                plot.save(os.path.join(self.save_path, 'iter_{}.png'.format(it)))

        return

    def _resume_from_dir(self):
        """ resume search from a previous iteration """
        import glob

        archive = []
        for file in glob.glob(os.path.join(self.resume, "net_*.subnet")):
            arch = json.load(open(file))
            pre, ext = os.path.splitext(file)
            stats = json.load(open(pre + ".stats"))
            archive.append((arch, 100 - stats['top1'], stats[self.sec_obj]))

        return archive

    def _evaluate(self, archs, it):
        gen_dir = os.path.join(self.save_path, "iter_{}".format(it))
        python_order = prepare_eval_folder(
            gen_dir, archs, self.gpu, self.n_gpus, data=self.data, dataset=self.dataset,
            n_classes=self.n_classes, supernet_path=self.supernet_path,
            num_workers=self.n_workers, valid_size=self.vld_size,
            trn_batch_size=self.trn_batch_size, vld_batch_size=self.vld_batch_size,
            n_epochs=self.n_epochs, test=self.test, latency=self.latency, verbose=False)

        subprocess.call("sh {}/run_bash.sh".format(gen_dir), shell=True)

        top1_err, complexity = [], []

        for i in range(len(archs)):
            try:
                stats = json.load(open(os.path.join(gen_dir, "net_{}_stats.txt".format(i))))
            except FileNotFoundError:
                # just in case the subprocess evaluation failed
                # 查找不到 stats 文件 则直接运行某条指令
                run_times = 0
                while True:
                    # subprocess.call('ls', shell=True)
                    subprocess.call(python_order[i], shell=True)
                    try:
                        stats = json.load(open(os.path.join(gen_dir, "net_{}_stats.txt".format(i))))
                        break
                    except Exception as e:
                        run_times += 1
                        print(f'fail-{run_times}-{e}')
                    if run_times >= 3:  # 执行 3 次还未出现结果文件则 赋予最差结果
                        stats = {'acc': 0, self.sec_obj: 1}  # makes the solution artificially bad so it won't survive
                        # store this architecture to a separate in case we want to revisit after the search
                        os.makedirs(os.path.join(self.save_path, "failed"), exist_ok=True)
                        shutil.copy(os.path.join(gen_dir, "net_{}_subnet.txt".format(i)),
                                    os.path.join(self.save_path, "failed", "it_{}_net_{}".format(it, i)))
                        break

            top1_err.append(1 - stats['acc'])  # 错误率 = 100 - 正确率
            complexity.append(stats[self.sec_obj])

        # evaluate 运行结果排序
        results = []
        result_log = gen_dir + "/iter_{}".format(it) + '_result.log'
        with open(result_log, 'r') as f:
            for line in f:
                results.append(line.strip('\n'))
        results = [x.strip() for x in results if x.strip() != '']
        results.sort(key=lambda x: int(re.findall(r'net_(\d+)_subnet.txt', x)[0]))
        with open(result_log, 'w') as f:
            for i in results:
                f.write(f'{i}\n')

        return top1_err, complexity

    def _fit_acc_predictor(self, archive):
        inputs = np.array([self.search_space.encode(x[0]) for x in archive])  # arch samples
        targets = np.array([x[1] for x in archive])  # acc
        assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

        acc_predictor = get_acc_predictor(self.predictor, inputs, targets)  # input是模型 targets是top-1错误率

        return acc_predictor, acc_predictor.predict(inputs)  # 返回精度预测器 和 预测器预测出来的精度值

    def _next(self, archive, predictor, K):
        """ searching for next K candidate for high-fidelity evaluation (lower level) """

        # the following lines corresponding to Algo 1 line 10 / Fig. 3(b) in the paper
        # get non-dominated architectures from archive
        F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # non-dominated arch bit-strings
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(
            self.search_space, predictor, self.sec_obj,
            {'n_classes': self.n_classes, 'model_path': self.supernet_path})

        # initiate a multi-objective solver to optimize the problem
        method = get_algorithm(
            "nsga2", pop_size=40, sampling=nd_X,  # initialize with current nd archs
            crossover=get_crossover("int_two_point", prob=0.9),
            mutation=get_mutation("int_pm", eta=1.0),
            eliminate_duplicates=True)

        # kick-off the search
        res = minimize(
            problem, method, termination=('n_gen', 20), save_history=True, verbose=True)

        # check for duplicates
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])

        # the following lines corresponding to Algo 1 line 11 / Fig. 3(c)-(d) in the paper
        # form a subset selection problem to short list K from pop_size
        indices = self._subset_selection(res.pop[not_duplicate], F[front, 1], K)
        pop = res.pop[not_duplicate][indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # decode integer bit-string to config and also return predicted top1_err
        return candidates, predictor.predict(pop.get("X"))

    @staticmethod
    def _subset_selection(pop, nd_F, K):
        problem = SubsetProblem(pop.get("F")[:, 1], nd_F, K)
        algorithm = GA(
            pop_size=100, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        res = minimize(
            problem, algorithm, ('n_gen', 60), verbose=False)

        return res.X

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        # calculate hypervolume on the non-dominated set of F
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = get_performance_indicator("hv", ref_point=ref_point).calc(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self, search_space, predictor, sec_obj='flops', supernet=None):
        super().__init__(n_var=46, n_obj=2, n_constr=0, type_var=np.int)

        self.ss = search_space
        self.predictor = predictor
        self.xl = np.zeros(self.n_var)
        self.xu = 2 * np.ones(self.n_var)
        self.xu[-1] = int(len(self.ss.resolution) - 1)
        self.sec_obj = sec_obj
        self.lut = {'cpu': 'data/i7-8700K_lut.yaml'}

        # supernet engine for measuring complexity
        self.engine = OFAEvaluator(
            n_classes=supernet['n_classes'], model_path=supernet['model_path'])

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)

        top1_err = self.predictor.predict(x)[:, 0]  # predicted top1 error

        for i, (_x, err) in enumerate(zip(x, top1_err)):
            config = self.ss.decode(_x)
            subnet, _ = self.engine.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d']})
            info = get_net_info(subnet, (3, config['r'], config['r']),
                                measure_latency=self.sec_obj, print_info=False, clean=True, lut=self.lut)
            f[i, 0] = err
            f[i, 1] = info[self.sec_obj]

        out["F"] = f


class SubsetProblem(Problem):
    """ select a subset to diversify the pareto front """

    def __init__(self, candidates, archive, K):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, type_var=np.bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # append selected candidates to archive then sort
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])))
            f[i, 0] = np.std(np.diff(tmp))
            # we penalize if the number of selected candidates is not exactly K
            g[i, 0] = (self.n_max - np.sum(_x)) ** 2

        out["F"] = f
        out["G"] = g


def main(args):
    engine = EMsNAS(vars(args))
    engine.search()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume search from a checkpoint')
    parser.add_argument('--sec_obj', type=str, default='flops',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--iterations', type=int, default=30,
                        help='number of search iterations')
    parser.add_argument('--n_doe', type=int, default=128,
                        help='initial sample size for DOE')
    parser.add_argument('--n_iter', type=int, default=8,
                        help='number of architectures to high-fidelity eval (low level) in each iteration')
    parser.add_argument('--predictor', type=str, default='rbf',
                        help='which accuracy predictor model to fit (rbf/gp/cart/mlp/as)')
    parser.add_argument('--n_gpus', type=int, default=8,
                        help='total number of available gpus')
    parser.add_argument('--gpu', type=int, default=1,
                        help='number of gpus per evaluation job')
    parser.add_argument('--data', type=str, default='/mnt/datastore/ILSVRC2012',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='number of classes of the given dataset')
    parser.add_argument('--supernet_path', type=str, default='../experiment/evo_croasen/super_net/super_model_best.pth.tar',
                        help='file path to supernet weights')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='number of workers for dataloader per evaluation job')
    parser.add_argument('--vld_size', type=int, default=None,
                        help='validation set size, randomly sampled from training set')
    parser.add_argument('--trn_batch_size', type=int, default=128,
                        help='train batch size for training')
    parser.add_argument('--vld_batch_size', type=int, default=200,
                        help='test batch size for inference')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='number of epochs for CNN training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluation performance on testing set')

    # add my cfg
    parser.add_argument('--bmname', dest='bmname',
                        help='Name of the benchmark dataset')
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
    parser.add_argument('--num_workers', dest='num_workers', type=int,
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
                        help='random generate graph size', default=False)
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
                        help='the smallest num of msg of a graph', default=50)  # 强化学习 每个步骤取一个图 构成这个图报文最小的条数
    parser.add_argument('--msg_biggest_num', type=int,
                        help='the biggest num of msg of a graph', default=300)  # 强化学习 每个步骤取一个图 构成这个图报文最大的条数

    parser.add_argument('--Di_graph', type=int,
                        help='Whether is Di-graph', default=1)  # 是否是有向图 默认为有向图

    parser.set_defaults(max_nodes=81,
                        feature_type='default',
                        datadir='../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/',
                        lr=0.001,
                        clip=2.0,
                        batch_size=64,
                        num_epochs=100,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=8,
                        # input_dim=10,
                        hidden_dim=20,  # hidden dim
                        output_dim=20,  # embedding dim
                        num_classes=2,  # 分两类 正常报文 异常报文
                        num_gc_layers=3,
                        dropout=0.0,
                        bmname='Pre_train',
                        )
    cfgs = parser.parse_args()
    main(cfgs)

