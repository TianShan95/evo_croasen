from codebase.data_providers.imagenet import *

from myofa.can_codebase.data_providers.chc_provider import *

from myofa.can_codebase.run_manager.run_manager import *
from logger.logger import logger
import pickle
from myofa.can_codebase.data_providers.randGraphData import RandGraphData
from myofa.can_codebase.data_providers.staticGraphData import OnlyGraphData
import myofa.can_codebase.data_providers.load_graph_origin_dataset_file as load_graph_origin_dataset_file
import networkx as nx
from myofa.can_codebase.data_providers.coarsen_pooling_with_last_eigen_padding import Graphs as gp


class ImagenetRunConfig(RunConfig):

    def __init__(self, n_epochs=1, init_lr=1e-4, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=128, test_batch_size=512, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224,
                 data_path='/mnt/datastore/ILSVRC2012',
                 **kwargs):
        super(ImagenetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )
        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.imagenet_data_path = data_path

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.imagenet_data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class ChcRunConfig(RunConfig):

    def __init__(self, n_epochs=1, init_lr=1e-4, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='CarHackingChallenge', train_batch_size=128, test_batch_size=512, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224,
                 # data_path='../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train',
                 dynamic_static='D',
                 csv_num='1_2',
                 pool_sizes='10',
                 normalization=True,
                 randGen=True,
                 seed=1,
                 gs=200,
                 args=None,
                 weight_dacay=None,
                 **kwargs):
        super(ChcRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )
        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.chc_data_path = args.datadir
        self.seed = seed
        self.weight_dacay = weight_dacay
        self.args = args

        # dynamic_static_list = args.ds.split('_')  # D or S
        # csv_num_list = args.csv_num.split('_')
        if args.data_PreMerge:  # 使用PreMerge文件夹数据
            data_path = args.datadir + 'MergeTrainSub_D/'
            data_out_dir = data_path + 'MergeTrainSub_D_0_1_2_sub_processed/ps_' + args.pool_sizes
            # 标记 邻接矩阵是否 拉普拉斯 归一化
            data_out_dir = data_out_dir + '_nor_' + str(bool(args.norm)) + '_' \
            # 是否是随机生成 还是 固定大小
            if args.randGen:
                data_out_dir = data_out_dir + 'random_' + str(args.seed) + '/'
            else:
                data_out_dir = data_out_dir + str(args.gs) + '/'

            if args.randGen:
                graph_list_file_name = f'{data_out_dir}/graphs_list_Di_{str(args.Di_graph)}_{args.msg_smallest_num}_{args.msg_biggest_num}.pth'
                dataset_file_name = f'{data_out_dir}/dataset_Di_{str(args.Di_graph)}_{args.msg_smallest_num}_{args.msg_biggest_num}.pth'  # 原始
            else:
                graph_list_file_name = f'{data_out_dir}/graphs_list_Di_{str(args.Di_graph)}.pth'
                dataset_file_name = f'{data_out_dir}/dataset_Di_{str(args.Di_graph)}.pth'

        else:
            # 若没有 dataset 则生成 dataset
            # 定义数据输出文件夹
            data_path = args.datadir + '0_Training/Pre_train'
            # data_path += args.bmname + '_'
            for i in args.ds:
                data_path += i + '_'  # 表明是 车辆动态报文 或者 车辆动态报文
            for i in args.csv_num:
                data_path += str(i) + '_'

            data_out_dir = data_path + 'processed/ps_' + args.pool_sizes
            # 标记 邻接矩阵是否 拉普拉斯 归一化
            data_out_dir = data_out_dir + '_nor_' + str(bool(args.norm)) + '_' \
                # 是否是随机生成 还是 固定大小
            if args.randGen:
                data_out_dir = data_out_dir + 'random_' + str(args.seed) + '/'
            else:
                data_out_dir = data_out_dir + str(args.gs) + '/'


            if args.randGen:
                graph_list_file_name = f'{data_out_dir}/graphs_list_Di_{str(args.Di_graph)}_{args.msg_smallest_num}_{args.msg_biggest_num}.pth'
                dataset_file_name = f'{data_out_dir}/dataset_Di_{str(args.Di_graph)}_{args.msg_smallest_num}_{args.msg_biggest_num}.pth'  # 原始
            else:
                graph_list_file_name = f'{data_out_dir}/graphs_list_Di_{str(args.Di_graph)}.pth'
                dataset_file_name = f'{data_out_dir}/dataset_Di_{str(args.Di_graph)}.pth'

        logger.info(f'查找数据集的文件位置是 {data_out_dir}')
        # 若 数据文件夹 不存在 则新建
        if not os.path.exists(data_out_dir):
            os.makedirs(data_out_dir)

        if os.path.isfile(graph_list_file_name) and os.path.isfile(dataset_file_name):
            # if False:
            logger.info('Files exist, reading from stored files....')
            logger.info(f'Reading file from {data_out_dir}')
            # 输出信息到log文件
            # with open(log_out_file, 'a') as f:
            logger.info(f'Files exist, reading from stored files....')
            logger.info(f'Reading file from{data_out_dir}')

            with open(dataset_file_name, 'rb') as f1:
                # 原始图的文件
                graphs = pickle.load(f1)
                f1.close()
            with open(graph_list_file_name, 'rb') as f2:
                # 坍塌处理过的 图文件
                graphs_list = pickle.load(f2)
                f2.close()
            logger.info('Data loaded!')
        else:
            logger.info('No files exist, preprocessing datasets...')

            # 生成图数据集
            if args.randGen:
                logger.info('随机生成图大小')
                p = RandGraphData(args)
            else:
                logger.info('生成固定图大小')
                p = OnlyGraphData(args)

            #
            graphs = load_graph_origin_dataset_file.read_graphfile(p.output_name_suffix, args.Di_graph,
                                              max_nodes=args.max_nodes)  # 从数据集得到图对象
            logger.info(f'Data length before filtering: {len(graphs)}')

            dataset_copy = graphs.copy()  # 拷贝原始图

            len_data = len(graphs)
            graphs_list = []  # 存储 坍缩图
            pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
            logger.info(f'pool_sizes: {pool_sizes}')

            # 遍历每个图 得到坍塌图的池化矩阵
            # 把处理好的数据储存
            for i in range(len_data):
                adj = nx.adjacency_matrix(dataset_copy[i])
                # print('Adj shape',adj.shape)
                if adj.shape[0] < args.min_nodes or adj.shape[0] > args.max_nodes or adj.shape[0] != dataset_copy[i].number_of_nodes():
                    graphs.remove(dataset_copy[i])
                    # index_list.remove(i)
                else:
                    # print('----------------------', i, adj.shape)
                    number_of_nodes = dataset_copy[i].number_of_nodes()
                    # if args.pool_ratios is not None:
                    #     pool_sizes = []
                    #     pre_layer_number_of_nodes = number_of_nodes
                    #     for i in range(len(pool_ratios)):
                    #         number_of_nodes_after_pool = int(pre_layer_number_of_nodes*pool_ratios[i])
                    #         pool_sizes.append(number_of_nodes_after_pool)
                    #         pre_layer_number_of_nodes = number_of_nodes_after_pool

                    # print('Test pool_sizes:  ', pool_sizes)
                    coarsen_graph = gp(adj.todense().astype(float), pool_sizes)
                    # if args.method == 'wave':
                    coarsen_graph.coarsening_pooling(args.normalize)
                    graphs_list.append(coarsen_graph)  # 坍缩图
            logger.info(f'Data length after filtering: {len(graphs)}, {len(graphs_list)}')
            logger.info('Dataset preprocessed, dumping....')

            # logger.info(f'这是测试， 屏蔽一下几行，不进行数据集缓存')
            with open(dataset_file_name, 'wb') as f:
                pickle.dump(graphs, f)
            with open(graph_list_file_name, 'wb') as f:
                pickle.dump(graphs_list, f)
            logger.info('Dataset dumped!')

        self.graphs = graphs
        self.graphs_list = graphs_list

        logger.info('Using node labels')
        # 使用节点标签作为图特征
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = np.array(G.nodes[u]['label'])

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ChcDataProvider.name():
                DataProviderClass = ChcDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
               args=self.args, graphs=self.graphs, graphs_list=self.graphs_list, seed=self.seed
            )
        return self.__dict__['_data_provider']


def get_run_config(**kwargs):
    if kwargs['dataset'] == 'imagenet':
        run_config = ImagenetRunConfig(**kwargs)
    elif kwargs['dataset'] == 'chc':
        run_config = ChcRunConfig(**kwargs)
    else:
        raise NotImplementedError

    return run_config


