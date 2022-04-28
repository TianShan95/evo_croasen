# 把 can 数据 处理为 图数据
# 固定长度{300}的 can 数据 构成 一个图
# 本文件独立运行 用来根据 原始can报文数据 生成 图网络数据集
import pandas as pd
import os
import copy
import collections
import logging
import random
from logger.logger import logger


# 处理 数据集 Car_Hacking_Challenge_Dataset_rev20Mar2021
# 20220310 增加 动态图功能 给定随机种子
class RandGraphData:
    def __init__(self, args):
        '''
        :param args:
        '''

        self.args = args


        # can_csv_dir 是数据集所在文件夹 和 数据集文件前缀 Pre_train
        # ../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train
        can_csv_dir = args.datadir + '0_Training/Pre_train'
        # self.fixed_len = args.gs  # 指定 每个图需要的 can 数据长度
        col_names = ['Arbitration_ID', 'Class']
        dataset_suffix_list = ['A', 'graph_indicator', 'graph_labels', 'node_labels']

        # Car_Hacking_Challenge_Dataset_rev20Mar2021 数据集
        # 读 csv文件数据
        output_ds_dir = can_csv_dir  # 输出数据集的文件夹名字
        frames_name = list()  # 可能存在多文件 合并情况
        frames = list()
        dataset_file_suffix = '_'  # 数据集文件的前缀
        # if args.dataset_name == 'Car_Hacking_Challenge_Dataset_rev20Mar2021':
        for i in args.ds:  # 遍历完 动态 和 静态
            read_can_csv_dir_suffix = can_csv_dir + '_' + i
            output_ds_dir += '_' + i
            dataset_file_suffix += i + '_'
            for j in args.csv_num:
                read_can_csv_dir = read_can_csv_dir_suffix + '_' + str(j) + '.csv'
                output_ds_dir += '_' + str(j)
                dataset_file_suffix += str(j) + '_'
                # frames.append(pd.read_csv(read_can_csv_dir, usecols=col_names))
                # 把文件名 存起来
                frames_name.append(read_can_csv_dir)
        # 判断文件夹 是否存在
        output_ds_dir = output_ds_dir + '_dataset/'  # 输出数据集路径 放在 与 原始csv 同路径下
        #
        self.output_name_suffix = f'{output_ds_dir}{os.path.basename(can_csv_dir)}{dataset_file_suffix}' \
                                  f'random_{args.seed}_{args.msg_smallest_num}_{args.msg_biggest_num}'  # 生成数据集的文件名前缀
        # 如果是 data_PreMerge 还需要添加 submission 文件夹中的数据
        if args.data_PreMerge:  # 使用PreMerge文件夹数据
            frames_name.append(f'{args.datadir}/1_Submission/Pre_submit_D.csv')
            output_ds_dir = f'{args.datadir}/MergeTrainSub_D/MergeTrainSub_D_0_1_2_sub'
            output_ds_dir = output_ds_dir + '_dataset/'  # 输出数据集路径 放在 与 原始csv 同路径下
            self.output_name_suffix = f'{output_ds_dir}MergeTrainSub{dataset_file_suffix}sub_' \
                                      f'random_{args.seed}_{args.msg_smallest_num}_{args.msg_biggest_num}'  # 生成数据集的文件名前缀

        if args.regen_dataset:
            regen = True
        else:
            regen = False  # 是否需要重新生成数据集
            if os.path.exists(output_ds_dir):  # 如果有数据集的文件夹
                for d_suffix in dataset_suffix_list:
                    if os.path.isfile(f'{self.output_name_suffix}_{d_suffix}.txt'):
                        continue
                    else:
                        regen = True
                        break
            else:
                os.makedirs(output_ds_dir)
                regen = True

        # 需要重新生成
        if regen:
            logger.info('数据集不存在 正在构造数据集 ...')
            for csv_file in frames_name:
                frames.append(pd.read_csv(csv_file, usecols=col_names))
            self.df = pd.concat(frames)  # 得到全部数据

            # 配置 log 文件 输出关键信息到 log 文件
            LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
            DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
            # 数据集生成日志文件
            logging.basicConfig(filename=self.output_name_suffix + '_DSGenerate.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
            logging.info('\n' + '==='*10)
            logging.info(f'数据集前缀 {self.output_name_suffix}')
            logging.info(f'随机种子为 {args.seed} ')
            logging.info(f'全部报文长度为 {len(self.df)} 帧')
            # logging.info(f'共产生 {int(len(self.df)/args.gs)} 个图数据')
            # 调用数据生成
            random.seed(args.seed)
            self.get_a()
        else:
            logging.info('数据集存在 若重新生成请设置相应参数')


    def get_a(self):

        # 节点相关
        can_id_list = self.df.get("Arbitration_ID").values  # 取出 ID 得到numpy类型 数据
        can_type_list = self.df.get("Class").values  # 取出 can 数据 的 报文类型 数据

        # 每个图报文长度列表
        graph_len_list = []

        # 转换为 十六进制ID 为键 十进制编号(节点编号) 为键值 的 节点字典
        # 字典为有序字典
        hex2decNum_dict = collections.OrderedDict()
        # 存放 每个图的 十六进制ID 和 十进制编号的 字典 列表
        # 列表每个元素 代表了一个图的 十六进制ID 和 十进制编号的 对应关系
        hex2decNum_dict_list = list()
        # 记录每个图的节点个数
        record_every_graph_node_num = list()
        # print('###')
        # print(len(can_id_list))
        # 同一个图内 ID相同 转换到的 十进制(节点编号) 也相同
        # 不同图 即使ID相同 转换到的 十进制(节点编号) 不相同
        # 无论是否在一个图中 ID相同 节点标签相同
        msg_point = 0  # 报文指针 每读取一个报文 加1 指上个图到达的报文位置
        read_done = False  # 标志 数据是否读取完成
        while True:
            graph_len = random.randint(self.args.msg_smallest_num, self.args.msg_biggest_num)

            for i in range(graph_len):  # 逐个图编号节点 # 遍历所有的 can ID 转换为 十进制
                try:
                    if can_id_list[msg_point+i] not in hex2decNum_dict.keys():  # 如果 此十六进制 ID 在此图中 未出现过 则需要新的节点编号
                        # ID 的节点编号是 从 1 开始 递增的
                        a = sum([len(h2d_dic) for h2d_dic in hex2decNum_dict_list]) + len(hex2decNum_dict) + 1
                        # print(f'a: {a}')
                        hex2decNum_dict[can_id_list[msg_point+i]] = a # 新出现一个 十六进制ID 则加入新键值对 键值 为 递增 节点标签 从 0 递增
                except IndexError:
                    # 如果最后一个图的数据 不够组成一个图 则舍弃
                    logger.info('can报文 文件读取完毕')
                    read_done = True
                    break
            # 文件读取完成 则退出 while循环
            if read_done:
                break
            # 此次报文读取无误 则加入此次的报文长度
            graph_len_list.append(graph_len)
            # logging.info(f'graphLen: {graph_len}')

            # 记录每个图的节点个数
            record_every_graph_node_num.append(len(hex2decNum_dict))
            # 读取完一个 图数据 把读取到的对应字典加入 列表
            hex2decNum_dict_list.append(copy.deepcopy(hex2decNum_dict))
            # 清空字典
            hex2decNum_dict.clear()
            # 进行下一个图的 遍历
            msg_point += graph_len

        # 输出 全部报文 共产生多少个不同编号的节点
        logging.info(f'全部报文共产生 {sum(record_every_graph_node_num)} 个不同编号节点')
        # 输出节点最多/最少图的节点个数到 日志文件
        logging.info(f'最少节点的图的节点个数为 {min(record_every_graph_node_num)}')
        logging.info(f'最多节点的图的节点个数为 {max(record_every_graph_node_num)}')
        # 检查转换的 图数据 个数
        logging.info(f'十六进制ID 转换 十进制 编号 共转换 {len(hex2decNum_dict_list)} 个图数据')

        # 遍历 hex2decNum_dict_list 里的每个字典 并输出文件
        # 每个字典的 键值(编号) 都应该是 递增 1 的
        # 输出文件 用来检验 遍历字典的顺序 是按照 字典的加入顺序进行的
        # 若第一列的数字 按行递增 1 则检验通过
        with open(self.output_name_suffix+'_check_node_num_dict.txt', 'w+') as f:
            for i in hex2decNum_dict_list:
                for key, value in i.items():
                    # print(str(value) + '  ' + str(key) + '\r\n')
                    f.write(str(value) + '  ' + str(key) + '\r\n')
        f.close()

        # 节点标签 虽然节点的编号不同 但是可能用有相同的节点标签
        # 无论是否在同一个图数据中 节点ID相同 节点标签相同
        # 节点标签从 1 开始
        # 因为 节点的编号因为在不同的图中 所以相同 CAN ID 的节点却有不同的节点编号 但是 他们的节点标签却是相同的
        # 遍历 全部数据 ===> CAN ID 和 节点标签的对应字典 键名：十六进制 字符串型 CAN ID 键值：十进制 int 型 节点标签
        with open(self.output_name_suffix+'_node_labels.txt', 'w+') as f:
            node_label_dict = collections.OrderedDict()
            # 得到每个报文ID 对应的 节点标签 节点标签从1开始
            for i in can_id_list:  # 遍历每帧报文数据的 ID
                if i not in node_label_dict.keys():
                    node_label_dict[i] = len(node_label_dict) + 1
            # 遍历每个图 根据节点编号和ID的对应关系 把里面的节点 都加上标签
            for graph_nodes_dic in hex2decNum_dict_list:  # 遍历每个图
                for node_num in graph_nodes_dic:  # 遍历图里的 节点编号 和 十六进制CAN ID 对；根据CAN ID 来写对应 的 节点标签
                    f.write(str(node_label_dict[node_num]) + '\n')
        f.close()

        # 输出 共有 多少种不同的节点标签
        logging.info(f'全部报文共有 {len(node_label_dict)} 种节点标签')

        # 检验步骤
        # 输出 十六进制报文ID 和 节点标签的对应关系
        # 如果节点标签列 每行递增 1 则通过检查
        with open(self.output_name_suffix+'_check_node_label_dict.txt', 'w+') as f:
            for key, value in node_label_dict.items():
                # print(str(value) + '  ' + str(key) + '\r\n')
                f.write(str(value) + '  ' + str(key) + '\r\n')
        f.close()


        # 获得 DS_A 边 文件
        graph_num = 0  # 遍历到的图的 位置
        graph_adj_list = list()  # 储存全部图的 连接边 每个元素表示一个图的连接关系
        adj_list = list()  # 图的边 每个元素是一个元祖 表示一条边
        len_adj_list = list()  # 记录每个图的边的个数

        for gl_index, graph_len in enumerate(graph_len_list):  # 遍历每个图的报文长度
            for i in range(graph_len):  # 逐个图 遍历 节点的连接关系
                try:
                    if (hex2decNum_dict_list[graph_num][can_id_list[sum(graph_len_list[:gl_index])+i]], hex2decNum_dict_list[graph_num][can_id_list[sum(graph_len_list[:gl_index])+i+1]]) not in adj_list:  # 如果边没在 列表中 则 添加入 列表
                        # 每个边是一个 tuple
                        adj_list.append((hex2decNum_dict_list[graph_num][can_id_list[sum(graph_len_list[:gl_index])+i]], hex2decNum_dict_list[graph_num][can_id_list[sum(graph_len_list[:gl_index])+i+1]]))

                except KeyError:
                    # print('%%%')
                    # print(f'i: {i} graph_len: {graph_len}')
                    # 如果最后一个图的数据 不够组成一个图 则舍弃
                    if i == graph_len - 1:
                        continue
                    else:
                        logging.info(f'KeyError 边 构造完毕 共构造了 {graph_num} 个图')
                        logger.info(f'KeyError 边 构造完毕 共构造了 {graph_num} 个图')
                        read_done = True
                        break
                except IndexError:
                    logging.info(f'IndexError 边 构造完毕 共构造了 {graph_num} 个图')
                    logger.info(f'IndexError 边 构造完毕 共构造了 {graph_num} 个图')
                    read_done = True
                    break

            # 把每个图的 边个数 记录下来
            len_adj_list.append(len(adj_list))

            # 调试 有图的边数为0 的原因
            if len(adj_list) == 0:
                logging.info(f'第 {graph_num} 个图的边数 为 0'
                             f'对应原始报文 {sum(graph_len_list[:gl_index])} - {sum(graph_len_list[:gl_index])+graph_len} 帧'
                             f'对应这些帧的报文ID为: '
                             f'gl_index: {gl_index}'
                             f'graph_len: {graph_len} i: {i}'
                             f'len(graph_len_list): {len(graph_len_list)}')
                for canid in can_id_list[sum(graph_len_list[:gl_index]):sum(graph_len_list[:gl_index])+graph_len]:
                    logging.info(canid)

            # 读取完一个 图数据 把读取到的边列表 加入 列表
            # print(f'adj_list: \n{adj_list}')
            graph_adj_list.append(copy.deepcopy(adj_list))
            # 清空列表
            adj_list.clear()
            # 进行下一个图的 遍历
            graph_num += 1

        # 输出所有图中 共产生了多少条边
        logging.info(f'所有的图共生成 {sum(len_adj_list)} 条边')
        # 输出所有图中 边最多/最少的图的边的个数
        logging.info(f'最少边的图的边有 {min(len_adj_list)} 条')
        logging.info(f'最多边的图的边有 {max(len_adj_list)} 条')




        # 写入 边文件
        with open(self.output_name_suffix + '_A.txt', 'w+') as f:
            for adj_list_ in graph_adj_list:  # 遍历每个图的边列表
                for edge_ in adj_list_:
                    f.write(str(edge_[0]) + ', ' + str(edge_[1]) + '\n')
        f.close()


        # 获得 图表示文件 DS_graph_indicator
        # 虽然每个 图数据是固定长度的 can 报文数据 但是并不代表每个图的节点数相同
        # 可以使用 十六进制对应十进制的 字典来获取
        # 图标识 从 1 开始 有多少个图 就到 多少
        with open(self.output_name_suffix + '_graph_indicator.txt', 'w+') as f:
            for graph_index_, h2d_dic_ in enumerate(hex2decNum_dict_list):  # 遍历 每个图
                # print(len(h2d_dic_))
                for i in range(len(h2d_dic_)):  # 该图有多少个节点 每个节点都加上属于哪个图的标识
                    f.write(str(graph_index_ + 1) + '\n')
        f.close()

        # 输出 graph_indicator 文件共写入多少个图
        logging.info(f'graph_indicator 文件共写入 {graph_index_+1} 个图')

        # 获得 图标签文件
        # 图标签 正常报文 1 异常报文 2
        with open(self.output_name_suffix + '_graph_labels.txt', 'w+') as f:
            for graph_num in range(len(hex2decNum_dict_list)):  # 遍历所有图的报文
                for i in range(graph_len_list[graph_num]):  # 遍历一个图的 所有报文类型 只要出现 attack 类型 则报文类型 为 attack
                    node_label = 1  # 设定初始标签为 正常报文
                    if can_type_list[sum(graph_len_list[:graph_num])+i] == 'Attack':
                        node_label = 2
                        # 若报文里出现一帧异常报文 则该图的标签是 2
                        break
                # 遍历完一个图 写入 文件
                f.write(str(node_label) + '\n')
        logging.info(f'graph_label文件写入完毕 共 {graph_num+1} 个图')

        f.close()


# if __name__ == '__main__':
#
#     # csv_dir = '/Users/aaron/Hebut/征稿_图像信息安全_20211130截稿/源程序/图塌缩分类/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training'
#     # bmname = "Pre_train"
#     # csv_dir += bmname
#     from args import arg_parse
#
#     args = arg_parse()
#     p = RandGraphData(args)
    # p.get_a()

    # # 查看 _A 文件
    # df_a = pd.read_csv('/Users/aaron/PycharmProjects/DynamicCANGraphLen/graphModel/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/D_1_A.txt')
    # print(df_a.tail())
    #
    # # 查看 节点编号 和 CAN ID 对应文件
    # df_a = pd.read_csv('/Users/aaron/PycharmProjects/DynamicCANGraphLen/graphModel/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/D_1_check_node_num_dict.txt', sep='  ')
    # print(df_a.tail())