import os
import re
import io


class Log2fitness:
    def __init__(self, log_file_dir):
        self.log_file_dir = log_file_dir

    def get_order_fit(self):
        log_name_list = adj_log_order(self.log_file_dir)  # 获得顺序的 log 文件 列表
        return get_log_best_acc(self.log_file_dir, log_name_list)  # 返回 按编号 顺序 排列 的 适应性 列表


def adj_log_order(log_file_dir):  # 调整 log 文件 按照 编号从小到大 的顺序排列
    '''
    :param log_file_dir: 本代种群训练 log 文件 路径 字符串
    :return: 按照编号大小顺序排列的 log 文件 列表
    '''
    files_in_log_dir = os.listdir(log_file_dir)
    log_files = []
    for file in files_in_log_dir:
        if os.path.splitext(file)[-1] == ".log":
            log_files.append(file)
    file_index_list = []
    # print('log 文件夹 中的 log 文件', log_files)
    for i in log_files:  # 遍历 log 文件夹里所有的 log 文件
        num = re.findall(r'(\d+)_', i, re.M)[0]
        file_index_list.append(int(num))
        # print('num: ', num)
    file_adj_list = []
    # print('file_index_list: ', file_index_list)
    for index, value in enumerate(file_index_list):
        file_adj_list.append(log_files[file_index_list.index(index)])
    # print('log_文件顺讯调整后:')
    # for i in file_adj_list:
    #     print(i)
    return file_adj_list


def get_log_best_acc(base_dir, log_file_list):
    '''
    :param base_dir: log文件 的 基础 文件夹
    :param log_file_list: log文件名 列表
    :return: log 列表中 每个 log 文件的 浮点型 评价指标(识别精度)
    '''

    p0_best_acc = []
    for i in log_file_list:
        # print(i)
        with io.open(base_dir + '/' + i, 'rt', encoding='UTF-8') as fd:
            for line in fd:
                if "best_acc" in line:
                    best_acc = re.findall(r"= (.*)", line, re.M)[0]
                    print(i, best_acc)
                    p0_best_acc.append(best_acc)
    return list(map(float, p0_best_acc))


def log_dir2fitness_list(log_dir):
    '''
    :param log_dir: log 路径
    :return:  按照 编号 排序 的 适应性列表
    '''
    return get_log_best_acc(log_dir, adj_log_order(log_dir))



if __name__ == '__main__':
    base_dir1 = 'popularization/1209_01_info/log_00000/'
    log_name_list1 = adj_log_order('popularization/1209_01_info/log_00000')
    get_log_best_acc(base_dir1, log_name_list1)