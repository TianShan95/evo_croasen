import os, sys, random
import platform
import numpy as np
import torch
import time
import logging
import traceback
# 工程参数

import argparse
# 强化学习
from RLModel.model.td3 import TD3
# 图网络
from graphModel.rltask import Task
# 功能函数
from utils.utils import setup_seed, random_list
# from utils.packResult import packresult
# from utils.sendMail import send_email
from logger.logger import logger
# 警告处理
import warnings
warnings.filterwarnings('ignore')  # 忽略警告
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from utils.WrapperModel import Wrapper as Model_Wrapper

# 参数初始化
# prog_args = arg_parse()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device: ', device)



'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''
'''
每个 can 长度的选项都是 一个维度
每次识别 can 报文的长度 就是 强化学习网络的输出维度
'''
global sys_cst_time
global train_times, avg_Q1_loss, avg_Q2_loss  # 系统时间


def main(prog_args):
    # 设置随机种子 方便实验复现
    setup_seed(prog_args.seed)

    # 状态维度
    # 这里的状态表示向量分别是 第一次图卷积操作（做了三次卷积 每次卷积产生 20 维向量） 图塌缩后 第二次卷积 同样是 产生 3 * 20 维
    # state_dim = ((prog_args.num_gc_layers - 1) * prog_args.hidden_dim + prog_args.output_dim) * 2  # 60 * 2
    state_dim = prog_args.state_dim
    # 动作维度
    action_dim = prog_args.msg_biggest_num - prog_args.msg_smallest_num + 1  # 每个图可选取报文长度的范围 动作空间 左闭右闭 [200, 500]

    # 获取当地时间
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    global sys_cst_time, train_times, avg_Q1_loss, avg_Q2_loss
    train_times = 0
    avg_Q1_loss = 0
    avg_Q2_loss = 0

    # 记录本次实验的输入参数
    with open('../experiment/exp-record.txt', 'a') as f:
        f.write(time_mark + '\t' + '\t'.join(sys.argv) + '\n')
        f.close()
    # 定义此次实验的 log 文件夹
    log_out_dir = prog_args.out_dir + '/' + 'Rl_' + time_mark + '_multiDim_log/'
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir, exist_ok=True)

    # 如果系统是 linux 则对系统时区进行设置
    # 避免日志文件中的日期 不是大陆时区
    # if platform.system().lower() == 'linux':
    #     print("linux")
    #     os.system("cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime")
    #     sys_cst_time = os.popen('date').read()
    #     print(f'系统时间: {sys_cst_time}')
    # 授时之后 系统时间更改 但是python获取的时间会有延迟
    # 验证时间 python 取到的时间是否和系统相符
    # timestr = 'Fri Feb 25 17:35:08 CST 2022'
    # while True:
    #     # python时间 和 系统时间 同步 退出
    #     if abs(time.mktime(time.strptime(sys_cst_time.strip('\n'), '%a %b %d %H:%M:%S CST %Y')) - time.time()) < 120:
    #         break
    #     time.sleep(0.5)
    # print(f'python 时间同步完成')

    agent = TD3(state_dim, action_dim, 1, prog_args, log_out_dir)  # 创建 agent 对象

    # tensorboard 可视化 actor 和 critic
    Wrapper = Model_Wrapper(agent.actor, agent.critic_1)
    agent.writer.add_graph(Wrapper, [torch.zeros([1,  prog_args.state_dim]).to(device), torch.zeros([1, prog_args.msg_biggest_num - prog_args.msg_smallest_num + 1]).to(device)])

    # 累加奖励
    ep_r = 0
    # 实例化 图任务
    graph_task = Task(prog_args, device)
    # pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]

    # 定义 并创建 log 文件
    log_out_file = log_out_dir + 'Rl_' + time_mark + '.log'
    # 配置日志 输出格式
    handler = logging.FileHandler(log_out_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # log 写入 参数
    logger.info('\t'.join(sys.argv))  # 写入运行程序时输入的参数
    logger.info(f'输出 log 文件路径: {log_out_file}')
    logger.info(f'{prog_args}')
    # logger.info(f'图模型- 学习率: {prog_args.graph_lr}')
    logger.info(f'强化学习学习率: {prog_args.reforce_lr}')

    error = None  # 表示实验是否发生异常
    retrain = False  # 表示模型是否是从开开始训练 False: 从头训练 True: 继续训练

    if prog_args.mode == 'test':

        # 载入 模型
        agent.load()
        # 记录 图模型 执行 步数
        graph_step = 0
        # for i in range(prog_args.train_epoch):
        # 随机获取 初始状态
        # 第一次随机 图的长度 [50-300] 闭空间 给出强化学习的 初始 state
        graph_len_list = random.randint(prog_args.msg_smallest_num, prog_args.msg_biggest_num)

        state, _, _ = graph_task.benchmark_task_val(prog_args.feat, graph_len_list)

        while True:
            action = agent.select_action(state)  # 从 现在的 状态 得到一个动作 维度是 报文长度可选择数量
            # 图操作 步数 自增 1
            graph_step += 1
            # 下个状态 奖励 是否完成
            len_can = np.argmax(action) + prog_args.msg_smallest_num  # 得到 下一块 数据的长度
            next_state, reward, done = graph_task.benchmark_task_val(prog_args.feat, len_can)
            # 累加 奖励
            ep_r += reward
            # 更新 状态
            state = next_state

            # 数据读取完毕
            # if done:
            #     agent.writer.add_scalar('ep_r', ep_r, global_step=i)
            #     if i % args_RL.print_log == 0:
            #         print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
            #     ep_r = 0
            #     break

    elif prog_args.mode == 'train':
        logger.info("====================================")
        logger.info("Collection Experience...")
        logger.info("====================================")

        # 若指定强化学习 载入模型参数 则 载入
        if prog_args.load:
            logger.info(f'加载模型 {prog_args.model_load_dir}')
            agent.load()  # 载入模型
        else:
            logger.info(f'模型从头开始训练\n')

        # 不需要每个 epoch 都重新赋值的变量
        last_val_acc = 0  # 记录上一轮的验证精度 如果精度上升则保存模型
        val_acc = 0  # 验证精度
        train_acc = 0  # 训练精度

        # try:
        for i in range(prog_args.train_epoch):  # epoch

            # 第一次随机 图的长度 [200-500] 闭空间 并且是batchsize个给出强化学习的 初始 state
            graph_len_list = random_list(prog_args.msg_smallest_num, prog_args.msg_biggest_num, prog_args.graph_batchsize)
            # 随机获取 初始状态 next_state, reward, train_done, val_done, label, pred, graph_loss
            state, _ , _, _, _, _, _ = graph_task.benchmark_task_val(prog_args.feat, graph_len_list)
            # print(f'随机得到的状态是 {state}')
            # 记录 图模型 执行 步数
            step = 0
            graph_train_step = 0
            graph_val_step = prog_args.graph_batchsize  # 因为模型读取train数据读完，未跳出读取 val 数据，所以需要赋予 batchsize 大小的初值
            # 记录正确预测的 报文 个数
            pred_train_correct = 0
            pred_val_correct = 0
            train_done = False
            states_np = None

            # states_np = np.array([], [])  # 存储每次图卷积网络输出的状态向量

            while True:
                # 强化学习网络
                # 串行把 batchsize 大小的数据输入 强化学习 得到batchsize大小个 can 数据长度
                len_can_list = []
                actions = []
                actual_select = []
                for singleCan in range(prog_args.graph_batchsize):
                    agent.actor.eval()
                    action = agent.select_action(state[singleCan], p=True)  # 从 现在的 状态 得到一个动作 报文长度可选择数量
                    actual_select.append(np.argmax(action) + prog_args.msg_smallest_num)

                    # agent.writer.add_graph(Wrapper, [torch.unsqueeze(state[singleCan], dim=0), torch.unsqueeze(torch.from_numpy(action).to(device), dim=0)])

                    # action = action + np.random.normal(0, prog_args.exploration_noise, size=action.shape[0])
                    # action = action.clip(-1, 1)

                    # len_can = 0
                    # 选取 前5 个最大的可能里 选择报文数最大的
                    # if prog_args.choice_graph_len_mode == 0:
                    #     len_can = np.argmax(action) + prog_args.msg_smallest_num  # 得到 下一块 数据的长度
                    if np.random.uniform() > prog_args.epsilon:  # choosing action
                        len_can = np.random.randint(prog_args.msg_smallest_num, prog_args.msg_biggest_num)
                    else:
                        # len_can = np.argmax(action) + prog_args.msg_smallest_num  # 得到 下一块 数据的长度
                        # np.random.choice 是左闭右开 所以加 1
                        len_can = np.random.choice(prog_args.msg_biggest_num - prog_args.msg_smallest_num + 1, 1, action.tolist())[0] + prog_args.msg_smallest_num

                    # elif prog_args.choice_graph_len_mode == 1:
                    #     # 选取 前5 个最大的可能里 选择报文数最大的
                    #     len_can = max(action.argsort()[::-1][0:5]) + prog_args.msg_smallest_num
                    # elif prog_args.choice_graph_len_mode == 2:
                    #     # 在 前 5 个最大的可能里 随机选择一个报文长度
                    #     alter = random.randint(0, 4)
                    #     len_can = action.argsort()[::-1][0:5][alter] + prog_args.msg_smallest_num

                    action_store = [1 if _ == (len_can - prog_args.msg_smallest_num) else 0 for _ in range(prog_args.msg_biggest_num - prog_args.msg_smallest_num + 1)]

                    # 输出 softmax 各个动作的概率
                    actions.append(action_store)

                    # 训练阶段
                    if not train_done:
                        # 图操作 训练步数 自增 1
                        graph_train_step += 1
                    else:
                        # 图操作 验证步数自增 1
                        graph_val_step += 1
                    step += 1  # 总 step

                    # if step % 1000 == 0:  #  每1000步输出 状态热力图
                    #     fig, ax = plt.subplots(figsize=(10, 10))
                    #     sns.heatmap(states_np, cmap="YlGnBu")
                    #     plt.savefig(log_out_dir + '/plt_state_%d_%d' % (i, step), dpi=300, bbox_inches='tight')
                    #     plt.clf()  # 更新画布
                    #     states_np = None


                    # 把神经网络得到的长度加入列表
                    len_can_list.append(len_can)

                next_state, reward, train_done, val_done, label, pred, graph_loss = graph_task.benchmark_task_val(prog_args.feat, len_can_list)

                # 最后结束
                if val_done:
                    # 把 train_done 和 val_done 置位
                    graph_task.origin_can_obj.train_done = False
                    graph_task.origin_can_obj.val_done = False

                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    # if i % args_RL.print_log == 0:
                    #     print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break  # break true

                # # 实时绘制 图神经网络的loss曲线
                # graph_task.history1.log((i, graph_train_step), graph_train_loss=graph_loss)
                # with graph_task.canvas1:
                #     graph_task.canvas1.draw_plot(graph_task.history1["graph_train_loss"])

                # 训练部分
                if not train_done:
                    # rewards = []
                    # push 经验
                    store_reward = []
                    for singleCan in range(prog_args.graph_batchsize):
                        # # 存入 经验
                        # if label[singleCan] == pred[singleCan]:
                        #     reward = abs(reward)
                        # else:
                        #     reward = -abs(reward)
                        if reward > 0.75:  # 预测准确率达 0.75 - 1
                            if label[singleCan] == pred[singleCan]:
                                store_reward.append(100)
                            else:
                                store_reward.append(-1)
                        elif reward > 0.5:  # 预测准确率达 0.5 - 0.75
                            if label[singleCan] == pred[singleCan]:
                                store_reward.append(10)
                            else:
                                store_reward.append(-100)
                        elif reward > 0.25:  # 预测准确率达 0.25 - 0.5
                            if label[singleCan] == pred[singleCan]:
                                store_reward.append(-10)
                            else:
                                store_reward.append(-100)
                        else :  # 预测准确率达 0.25
                            if label[singleCan] == pred[singleCan]:
                                store_reward.append(-50)
                            else:
                                store_reward.append(-100)

                        # 累加 奖励
                        ep_r += reward
                        # rewards.append(reward)
                        agent.memory.push((state[singleCan].cpu().data.numpy().flatten(),
                                           next_state[singleCan].cpu().data.numpy().flatten(),
                                           actions[singleCan], store_reward[-1], np.float(train_done)))
                        if len(agent.memory.storage) >= prog_args.capacity - 1:
                            train_times, avg_Q1_loss, avg_Q2_loss = agent.update(num_iteration=10)  # 使用经验回放 更新网络

                    # 计数训练时 预测正确的个数
                    for index, singlab in enumerate(label):
                        if singlab == pred[index]:
                            pred_train_correct += 1

                    # 得到训练精度
                    train_acc = pred_train_correct/graph_train_step
                    agent.writer.add_scalar('acc/train_acc', train_acc, global_step=graph_train_step)
                    agent.writer.add_scalar('Loss/graph_train_loss', graph_loss, global_step=graph_train_step)


                    # 结果写入 log
                    logger.info(f'epoch-train: {i:<3}; train-step: {graph_train_step:<6}; '
                                f'block_{graph_task.origin_can_obj.train_index}: {graph_task.origin_can_obj.train_order[graph_task.origin_can_obj.train_index]}; '
                                f'{graph_task.origin_can_obj.point}/{graph_task.origin_can_obj.data_train_block_len}; '
                                f'reward: {reward:<8.3f}; '
                                f'acc: {train_acc:<4.2f}; trainTimes: {train_times}; g_loss: {graph_loss:<8.6f}; '
                                f'avg_Q1_loss: {avg_Q1_loss:.2f}; avg_Q2_loss: {avg_Q2_loss:.2f}; ep_r: {ep_r:.2f}')
                    logger.info(f'actual selec: {actual_select}')
                    logger.info(f'len_can_list: {len_can_list}')
                    logger.info(f'labe: {label}')
                    logger.info(f'pred: {pred}')
                    logger.info(f'swrd: {store_reward}')

                # 验证部分
                else:

                    store_reward = []
                    for singleCan in range(prog_args.graph_batchsize):
                        # # 存入 经验
                        # if label[singleCan] == pred[singleCan]:
                        #     reward = abs(reward)
                        # else:
                        #     reward = -abs(reward)
                        # # 累加 奖励
                        # ep_r += reward
                        if reward > 0.75:  # 预测准确率达 0.75 - 1
                            if label[singleCan] == pred[singleCan]:
                                store_reward.append(100)
                            else:
                                store_reward.append(-1)
                        elif reward > 0.5:  # 预测准确率达 0.5 - 0.75
                            if label[singleCan] == pred[singleCan]:
                                store_reward.append(10)
                            else:
                                store_reward.append(-100)
                        elif reward > 0.25:  # 预测准确率达 0.25 - 0.5
                            if label[singleCan] == pred[singleCan]:
                                store_reward.append(-10)
                            else:
                                store_reward.append(-100)
                        else:  # 预测准确率达 0.25
                            if label[singleCan] == pred[singleCan]:
                                store_reward.append(-50)
                            else:
                                store_reward.append(-100)

                    # 计数训练时 预测正确的个数
                    for index, singlab in enumerate(label):
                        if singlab == pred[index]:
                            pred_val_correct += 1
                    # 得到验证精度
                    val_acc = pred_val_correct/graph_val_step

                    agent.writer.add_scalar('acc/val_acc', val_acc, global_step=graph_val_step)
                    agent.writer.add_scalar('Loss/graph_val_loss', graph_loss, global_step=graph_val_step)
                    # 结果写入 log
                    logger.info(f'epoch-val: {i:<3}; step: {graph_val_step:<6}; '
                                f'{graph_task.origin_can_obj.point}/{graph_task.origin_can_obj.data_val_len}; '
                                f'reward: {reward:<8.3f}; '
                                f'acc: {val_acc:<4.2f}; g_loss: {graph_loss:<8.6f}; ep_r: {ep_r:.2f}')
                    logger.info(f'actual selec: {actual_select}')
                    logger.info(f'len_can_list: {len_can_list}')
                    logger.info(f'labe: {label}')
                    logger.info(f'pred: {pred}')
                    logger.info(f'swrd: {store_reward}')

                # 更新 状态
                state = next_state

                # if graph_train_step < 100 or states_np is None:
                #     states_np = state.cpu().detach().numpy()
                # else:
                #     states_np = np.concatenate((states_np, state.cpu().detach().numpy()), axis=0)


                # # 保存 模型
                # if graph_step % args_RL.log_interval == 0:
                #     agent.save()
                #     break

            # 记录此次的训练精度 和 验证精度
            logger.info(f'epoch-{i}-over '
                        f'trian-times: {train_times} '
                        f'train_acc: {train_acc:<4.6f} '
                        f'val_acc: {val_acc:<4.6f}')
            # 跳出whileTrue 结束epoch 保存模型
            # 如果此次的验证精度上升则保存模型

            # 置位完成标识位
            graph_task.origin_can_obj.train_done = False
            graph_task.origin_can_obj.val_done = False

            if val_acc > last_val_acc:
                # 保存本次的验证精度
                last_val_acc = val_acc
                # 保存强化学习模型
                agent.save(i, str('%.4f' % val_acc), log_out_dir)
                # 保存图模型
                # graph_model_path = log_out_dir + 'epoch_' + str(i) + '_graph.pth.tar'
                # graph_model_para_path = log_out_dir + 'epoch_' + str(i) + '_graph_para.pth'
                # torch.save({'model' : graph_task.model, 'state_dict': graph_task.model.state_dict()}, graph_model_path)
                # torch.save(graph_task.model.state_dict(), graph_model_para_path)

            # # 结束一次 epoch 发送一次邮件 防止 colab 突然停止
            # content = f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())} END\n' \
            #           f'epoch: {i}\n'\
            #           f'retrain: {retrain}\n'
            # resultfile = packresult(log_out_dir[:-1], i)  # 1.传入log路径参数 去掉最后的 / 2. 训练结束的代数
            # send_email(prog_args.username, prog_args.password, prog_args.sender, prog_args.receivers,
            #            prog_args.smtp_server, prog_args.port, content, resultfile)

        # except Exception as e:  # 捕捉所有异常
        #     logger.info(f'发生异常 {e}')
        #     error = e
        #
        # finally:
        #     # 异常信息写入 log
        #     logger.warning(f'error: {error}')
        #     # 程序执行失败信息写入 log
        #     traceback.print_exc()
        #     logger.warning(f"执行失败信息: {traceback.format_exc()}")
            # 无论实验是否执行完毕 都把结果发送邮件
            # 跑完所有的 epoch 打包实验结果 返回带 .zip 的文件路径
            # print(f'正在打包结果文件夹  {log_out_dir}')
            # agent.save(i, log_out_dir)  # 保存 最新的模型参数
            # resultfile = packresult(log_out_dir[:-1], i)  # 1.传入log路径参数 去掉最后的 / 2. 训练结束的代数
            # print(f'打包完毕')
            # 发送邮件
            # print(f'正在发送邮件...')
            # content = f'platform: {prog_args.gpu_device}\n'\
            #           f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())} END\n' \
            #           f'retrain: {retrain}\n' \
            #           f'schedule: {graph_task.origin_can_obj.point}/{graph_task.origin_can_obj.data_total_len}\n' \
            #           f'error: {error}\n'

            # send_email(prog_args.username, prog_args.password, prog_args.sender, prog_args.receivers, prog_args.smtp_server, prog_args.port, content,resultfile)
            # print(f'发送邮件完毕')

            # # 如果是在 share_gpu 上运行的 把数据都拷贝到 oss 个人数据下
            # if prog_args.gpu_device == 'share_gpu':
            #     # 全部打包
            #     resultfile = packresult(log_out_dir[:-1], i, allfile=True)  # 1.传入log路径参数 去掉最后的 / 2. 训练结束的代数
            #     os.system(f"oss cp {resultfile} oss://backup/")
            #     print('关机...')
            #     os.system('/root/upload.sh')


    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments.')

    # 数据集相关
    parser.add_argument('--bmname', dest='bmname', help='Name of the benchmark dataset')

    # 强化学习相关
    parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--train_epoch', default=1, type=int)  # 训练代数
    parser.add_argument('--exploration_noise', default=0.1, type=float)  # 拓展噪音
    # parser.add_argument('--test_iteration', default=5, type=int)
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)

    # parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--capacity', default=3000, type=int)  # replay buffer size 1000
    parser.add_argument('--batch_size', default=100, type=int)  # mini batch size  100
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')

    # 定义 并创建 此次实验的 log 文件夹
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # parser.add_argument('--model_store_dir', default='../rl_model_store/' + time_mark + '_', type=str)
    parser.add_argument('--model_load_dir',
                        default='', type=str)

    parser.add_argument('--out_dir', type=str,
                        help='out_dir', default='../../experiment')  # 实验结果输出文件夹

    # optional parameters
    # parser.add_argument('--activation', default='Relu', type=str)
    # parser.add_argument('--log_interval', default=50, type=int)  # 打印 log 的间隔
    # parser.add_argument('--load', action="store_true",help='load rl model or not')  # load model  在训练时 强化学习是否加载模型
    parser.add_argument('-load', action='store_true', default=False)
    # parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work

    parser.add_argument('--policy_delay', default=2, type=int)
    # parser.add_argument('--policy_noise', default=0.2, type=float)  #噪声相关
    # parser.add_argument('--noise_clip', default=0.5, type=float)  # 噪声相关
    # parser.add_argument('--exploration_noise', default=0.1, type=float)  # 对于强化学习输出的 选择can长度加入噪声

    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
                        help='Ratio of number of graphs testing set to all graphs.')

    # 图网络部分
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')

    # parser.add_argument('--graph_model_path',
    #                     default='../experiment/randGen_42_Normlize_True_concat_1_20220312_211619_log/0.87_better_model__totalEpoch_300_epoch_132_ps_10_nor_1.pth',
    #                     type=str)
    # parser.add_argument('--graph_model_path',
    #                     default='',
    #                     type=str)
    parser.add_argument('--graph_batchsize', dest='graph_batchsize', type=int,
                        help='batchsize to train graph neural network')
    parser.add_argument('--graph_num_workers', dest='graph_num_workers', type=int,
                        help='Number of workers to load processData.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    # parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
    #                     help='Hidden dimension')
    # parser.add_argument('--output-dim', dest='output_dim', type=int,
    #                     help='Output dimension')
    # parser.add_argument('--num-classes', dest='num_classes', type=int,
    #                     help='Number of label classes')  # 图网络输出的 维度
    # parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
    #                     help='Number of graph convolution layers before each pooling')  # 卷积层数目
    # parser.add_argument('--dropout', dest='dropout', type=float,
    #                     help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--origin-can-datadir', dest='origin_can_datadir',
                        help='Directory where origin can dataset is located')  # can 数据的路径
    parser.add_argument('--pool_sizes', type=str,
                        help='pool_sizes', default='10')  # 分簇时 每个簇里节点的个数
    # parser.add_argument('--group_sizes', type=str,
    #                     help='group_sizes', default='10')  # 分簇时 分的簇的个数
    parser.add_argument('--num_pool_matrix', type=int,
                        help='num_pooling_matrix', default=1)

    parser.add_argument('--weight_decay', type=float,
                        help='weight_decay', default=0.0)

    # parser.add_argument('--choice_graph_len_mode', type=int,
    #                     help='choice graphLen mode', default=0)  # 0: 只选择最可能的维度 1: 从前5个中选最大 2: 从前5个中随机选

    parser.add_argument('--num_pool_final_matrix', type=int,
                        help='number of final pool matrix', default=0)
    # parser.add_argument('--normalize', type=int,
    #                     help='nomrlaized laplacian or not', default=0)
    # parser.add_argument('--pred_hidden', type=str,
    #                     help='pred_hidden', default='120')
    # parser.add_argument('--concat', type=int,
    #                     help='whether concat', default=1)  # 是否把每层卷积的特征向量都拼接起来 输出给图部分的预测层
    parser.add_argument('--feat', type=str,
                        help='which feat to use', default='node-label')
    parser.add_argument('--mask', type=int,
                        help='mask or not', default=1)
    parser.add_argument('--norm', type=str,
                        help='Norm for eigens', default='l2')
    parser.add_argument('--con_final', type=int,
                        help='con_final', default=1)
    parser.add_argument('--device', type=str,
                        help='cpu or cuda', default='cpu')

    # 使用 Car_Hacking_Challenge_Dataset_rev20Mar2021 数据集
    # 生成数据集需要修改的参数
    parser.add_argument('--dataset_name', type=str,
                        help='dynamic or static', default='Car_Hacking_Challenge_Dataset_rev20Mar2021')  # 0 or 1 or 2

    parser.add_argument('--ds', type=list,
                        help='dynamic or static', default=['D'])  # D or S 车辆动态报文 或者 车辆静止报文
    parser.add_argument('--csv_num', nargs='+', type=int,
                        help='csv num', default=[1, 2])  # 0 or 1 or 2  # csv文件标号
    parser.add_argument('--data_PreMerge', dest='data_PreMerge', action='store_const',
                        const=False, default=True,
                        help='Whether to use submission')
    parser.add_argument('--msg_smallest_num', type=int,
                        help='the smallest num of msg of a graph', default=200)  # 强化学习 每个步骤取一个图 构成这个图报文最小的条数
    parser.add_argument('--msg_biggest_num', type=int,
                        help='the biggest num of msg of a graph', default=500)  # 强化学习 每个步骤取一个图 构成这个图报文最大的条数

    # 实验结果发送邮件相关
    parser.add_argument('--username', type=str,
                        help='user name', default="976362661@qq.com")  # 用户名
    parser.add_argument('--password', type=str,
                        help='password', default="vsgbohogiqerbcji")  # 授权码
    parser.add_argument('--sender', type=str,
                        help='sender email', default='976362661@qq.com')  # 发送邮箱
    parser.add_argument('--receivers', type=str,
                        help='recervers email', default='ts_951117aaron@163.com')  # 接收邮箱
    parser.add_argument('--smtp_server', type=str,
                        help='smtp server', default="smtp.qq.com")  # smtp 服务器
    parser.add_argument('--port', type=int,
                        help='smtp port', default=465)  # smtp 端口

    # 显卡平台选择 涉及数据保存
    parser.add_argument('--gpu_device', type=str,
                        help='gpu platform', default="colab")  # colab or share_gpu
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')

    # 图网络学习率
    # parser.add_argument('--graph_lr', type=float, dest='graph_lr', help='learning rate of graph neural network')
    # 强化学习学习率
    parser.add_argument('--reforce_lr', type=float, dest='reforce_lr',
                        help='learning rate of reforcemet learning network')
    parser.add_argument('--epsilon', type=float, dest='epsilon', help='random exploration')

    # parser.add_argument('--Di_graph', type=int,
    #                     help='Whether is Di-graph', default=0)  # 是否是有向图 默认为有向图

    parser.add_argument('--state_dim', type=int,
                        help='Whether is Di-graph', default=40)  # 是否是有向图 默认为有向图

    parser.add_argument('--graph_cfg_file', type=str,
                        help='the file to config graph neural network', default="./.tmp/iter_30/net_3_subnet.txt")  # 图神经网络的配置文件路径

    parser.set_defaults(max_nodes=81,
                        feature_type='default',
                        # graph_lr=0.001,  # 0.001
                        reforce_lr=0.1,  # 0.1
                        clip=2.0,
                        train_ratio=0.8,
                        graph_batchsize=64,  # 一个图的选择是一个动作
                        train_epoch=20,
                        graph_num_workers=8,
                        input_dim=81,
                        # hidden_dim=20,
                        # output_dim=20,
                        num_classes=2,  # 正常报文和入侵报文
                        num_gc_layers=2,
                        # dropout=0.0,
                        bmname='Pre_train',
                        origin_can_datadir='../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/',

                        # train_epoch = 20,
                        # graph_batchsize = 64,
                        epsilon=0.9,
                        msg_smallest_num=200,
                        msg_biggest_num=500,
                        graph_model_path='../../experiment/evo_croasen_20220504_143455/iter_30/net_3/checkpoint/model_best.pth.tar',
                        # graph_lr = 0.01,
                        # reforce_lr = 30,
                        )

    main(parser.parse_args())
