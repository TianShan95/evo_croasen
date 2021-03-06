import pandas as pd
import networkx as nx
import copy
from random import shuffle
from logger.logger import logger
import os
import collections
import matplotlib.pyplot as plt


class OriginCanData:
    def __init__(self, args):
        col_name_list = ['Arbitration_ID', 'Class']  # 需要取出 文件 的 列名称

        can_csv_dir = args.origin_can_datadir + args.bmname  #  ../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train

        csv_names = list()
        if args.dataset_name == 'Car_Hacking_Challenge_Dataset_rev20Mar2021':
            for i in args.ds:  # 遍历完 动态 和 静态
                read_can_csv_dir_suffix = can_csv_dir + '_' + i
                for j in args.csv_num:
                    read_can_csv_dir = read_can_csv_dir_suffix + '_' + str(j) + '.csv'
                    # frames.append(pd.read_csv(read_can_csv_dir, usecols=col_names))
                    # 把文件名 存起来
                    csv_names.append(copy.deepcopy(read_can_csv_dir))

        if args.data_PreMerge:  # 如果需要加上 submission_D 部分
            csv_names.append(f'{os.path.dirname(os.path.dirname(args.origin_can_datadir))}/1_Submission/Pre_submit_D.csv')
        frames = list()
        for csv_file in csv_names:
            logger.info(f'数据csv文件: {csv_file}')
            frames.append(pd.read_csv(csv_file, usecols=col_name_list))  # 读取 can报文 csv源文件
        self.df = pd.concat(frames)  # 得到全部数据

        self.point = 0  # 记录 此次取出的can数据 结束 位置
        self.data_total_len = len(self.df)

        self.df_train = self.df.iloc[:int(self.data_total_len*args.train_ratio), :]
        # 把训练部分平均分为10份，打乱
        self.df_train_list = [self.df.iloc[:int(self.data_total_len*args.train_ratio*0.1), :],
                              self.df.iloc[int(self.data_total_len*args.train_ratio*0.1):int(self.data_total_len*args.train_ratio*0.2), :],
                              self.df.iloc[int(self.data_total_len*args.train_ratio*0.2):int(self.data_total_len*args.train_ratio*0.3), :],
                              self.df.iloc[int(self.data_total_len*args.train_ratio*0.3):int(self.data_total_len*args.train_ratio*0.4), :],
                              self.df.iloc[int(self.data_total_len*args.train_ratio*0.4):int(self.data_total_len*args.train_ratio*0.5), :],
                              self.df.iloc[int(self.data_total_len*args.train_ratio*0.5):int(self.data_total_len*args.train_ratio*0.6), :],
                              self.df.iloc[int(self.data_total_len*args.train_ratio*0.6):int(self.data_total_len*args.train_ratio*0.7), :],
                              self.df.iloc[int(self.data_total_len*args.train_ratio*0.7):int(self.data_total_len*args.train_ratio*0.8), :],
                              self.df.iloc[int(self.data_total_len*args.train_ratio*0.8):int(self.data_total_len*args.train_ratio*0.9), :],
                              self.df.iloc[int(self.data_total_len*args.train_ratio*0.9):int(self.data_total_len*args.train_ratio*1.0), :]]
        self.train_order = [x for x in range(10)]
        self.train_index = 0
        shuffle(self.train_order)
        logger.info(f'训练顺序: {self.train_order}')

        self.df_val = self.df.iloc[int(self.data_total_len*args.train_ratio):, :]

        # self.data_train_len = len(self.df_train)
        self.data_train_block_len = len(self.df_train_list[self.train_index])

        self.data_val_len = len(self.df_val)

        self.train_done = False
        self.val_done = False

        # self.args = args

        # print(self.df.head())
        # print(self.df.tail())

    def get_ds_a(self, len_can_list, graph_direction):

        # done = False
        # graph = None
        graphs = []
        for len_can in len_can_list:
            if self.train_done:
                df = self.df_val.iloc[self.point:self.point + len_can, :]  # 从验证can报文 取出 特定长度 的 验证 can报文
            else:
                df = self.df_train_list[self.train_order[self.train_index]].iloc[self.point:self.point+len_can, :]  # 从训练can报文 取出 特定长度 的 训练 can报文

            # 图相关
            graph_label = 0  # 标识图标签 0：正常报文 1：入侵报文

            # 边相关
            adj_list = list()  # 存储 每条 连接边
            edge_weight_list = list()  # 存储 每条边的权重 每重复连接一次 加1

            # 节点相关
            can_id_list = df.get("Arbitration_ID").values  # 取出 ID 得到numpy类型 数据
            can_type_list = df.get("Class").values  # 取出 can 数据 的 报文类型 数据

            # print(type(can_id))
            # print(f'canData length: {len(can_id_list)} range: {self.point} - {self.point+len_can-1}/{self.data_total_len}')
            # with open(log_out_file, 'a') as f:
            #     f.write(f'processing CAN {self.point}/{self.data_total_len}\n')
            # print(df)
            # print('***')

            # 转换为 十六进制ID 为键 十进制节点编号(从1开始) 为键值 的 节点字典
            hex2num_dict = dict()
            # print('###')
            # print(len(can_id))
            for i in range(len_can):
                try:
                    if can_id_list[i] not in hex2num_dict.keys():  # 如果 此十六进制 ID 未出现过
                        hex2num_dict[can_id_list[i]] = len(hex2num_dict) + 1  # 新出现一个 十六进制ID 则加入新键值对 键值 为 递增 节点标签 从 0 递增
                except IndexError:
                    # print(i)
                    # print('can报文 文件读取完毕')
                    # done = True
                    if not self.train_done:
                        # 训练阶段完成 置标志位 把报文指针置0
                        self.point = 0
                        if self.train_index >= 9:
                            self.train_index = 0  # 训练部分执行完毕 训练块索引 置位
                            self.train_done = True
                        else:  # 如果训练块没有结束 执行下一个块
                            self.train_index += 1
                            # 下一个块的 报文总帧数
                            self.data_train_block_len = len(self.df_train_list[self.train_index])
                        # 训练部分结束
                        # 重新获取 验证的报文长度
                        return self.get_ds_a(len_can_list, graph_direction)
                    else:
                        # 验证阶段完成 置标志位 把报文指针置0 以便下一个epoch进行
                        self.point = 0
                        self.val_done = True

                        # 重新打乱 训练集
                        shuffle(self.train_order)
                        logger.info(f'训练顺序: {self.train_order}')
                    break

            # print(f'节点数: {len(hex2num_dict)}')

            if not self.val_done:
                # 节点从0编号结束 打印一下 字典
                # print(f'节点ID和节点编号对应关系: {hex2num_dict}')
                # print(f'此图具有的节点个数为: {len(hex2num_dict)}')

                # 得到 图实例 所需要的数据
                for i in range(len_can):  # 遍历 本次 选出 的 报文数据
                    try:
                        # 图标签
                        if can_type_list[i] != 'Normal':  # 如果含有入侵数据 则 判定为 入侵报文
                            graph_label = 1

                        # 节点
                        # 图的节点标签 直接 从 节点十六进制 到 十进制 的字典得到

                        # 边 用节点编号表示的边
                        if (hex2num_dict[can_id_list[i]], hex2num_dict[can_id_list[i+1]]) not in adj_list:  # 如果边没在 列表中 则 添加入 列表
                            adj_list.append((hex2num_dict[can_id_list[i]], hex2num_dict[can_id_list[i+1]]))

                        # 以下代码为 考虑 边权重 情况
                        #     edge_weight_list.append(1)
                        # else:  # 边已经 被储存在 列表
                        #     print(f'节点 {i} {adj_list.index((can_id[i], can_id[i+1]))} 重复连接')
                        #     edge_weight_list[adj_list.index((can_id[i], can_id[i+1]))] += 1

                    except IndexError:
                        # print('发生读取异常 预想为读到了 canID最后一个')
                        # print(f'读取完毕: 此时的 i 为 {i}, 此时的 range(len_can) 为 {len_can}')
                        pass  # 当遍历到 数据 最后一个点 会触发异常 退出循环
                # print(f'边列表:\n {adj_list}')
                # print(f'边数 {len(adj_list)} \n')
                # print(edge_weight_list)
                # print(len(edge_weight_list))
                # print(f'节点编号{hex2num_dict.values()}')

                # 实例化 图
                # graph = nx.from_edgelist(adj_list)  # 从边 列表 构造 图
                if graph_direction:
                    graph = nx.from_edgelist(adj_list, create_using=nx.DiGraph())  # 从边 列表 构造 图
                else:
                    graph = nx.from_edgelist(adj_list)  # 从边 列表 构造 图

                graph.graph['label'] = graph_label
                # print(f'报文类型: {graph_label}')
                # logger.info(f'label: {graph_label}; canData length: {len(can_id_list):3}; schedule: {self.point:7}/{self.data_total_len}')

                # Plot the graph 可视化建立的 图实例
                # nx.draw(graph, with_labels=True, font_weight='bold')
                # plt.show()

                # 给每个节点编号 加 节点标签
                # 每个标签代表了一种 canID
                # 字典的键名表示 CAN ID 键值 表示 标签
                # 节点标签从 1 开始
                # 这里读到的节点的标签 需要和 在图坍缩是的 节点ID和标签的对应关系一致
                # 图坍缩 的 节点ID 和 标签 的对应关系 在同路径下的 node_label_dict.txt 文件里
                node_label_df = pd.read_csv('graphModel/processData/node_label_dict.txt', sep='  ', usecols=['label', 'id'], engine='python')
                result_dic = node_label_df.groupby('id')['label'].apply(list).to_dict()

                # 添加图标签 使用十进制的 图标签 的 one-hot 作为 图节点标签
                for u in graph.nodes():
                    node_label_one_hot = [0] * len(result_dic) # 图标签 种类数 个零 的 一个 列表 保持每次训练的一致性 把节点特征维度设为最大节点数
                    canId = [k for k, v in hex2num_dict.items() if v == u][0]  # 根据键值找到 canID
                    node_label = result_dic[canId][0]
                    node_label_one_hot[node_label-1] = 1  # u 是节点编号 且从 1 开始  节点标签也是从 1 开始
                    graph.nodes[u]['label'] = node_label_one_hot

                self.point += len_can

                # for u in graph.nodes():
                #     print(u)
                #     print(graph.nodes[u]['label'])

                # relabeling
                mapping = {}
                it = 0
                if float((nx.__version__)[:3]) < 2.0:
                    for n in graph.nodes():
                        mapping[n] = it
                        it += 1
                else:
                    for n in graph.nodes:
                        # print('###')
                        # print(G.nodes[n]['label'])
                        mapping[n] = it
                        it += 1

                graph = nx.relabel_nodes(graph, mapping)
                graphs.append(graph)


        return graphs, self.train_done, self.val_done

    def test(self, can_len):
        df = self.df.iloc[179346:179349, :]  # 取出 特定长度 的can报文
        print(df)
        print(len(self.df))


# if __name__ == '__main__':
#     origin_can_dir_ = '/Users/aaron/git_project/eigenpooling/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/' \
#                       '0_Preliminary/0_Training/Pre_train_D_1.csv'
#     p = OriginCanData(origin_can_dir_)
#     # for i in range(10000):
#     print(p.get_ds_a(20))
#     # p.test(2)
