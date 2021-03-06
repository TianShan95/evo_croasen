from tensorboardX import SummaryWriter
import torch
from RLModel.model.actor import Actor
from RLModel.model.critic import Critic
from RLModel.model.replayBuff import Replay_buffer

import torch.optim as optim
import torch.nn.functional as F
from logger.logger import logger
import hiddenlayer as hl


'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TD3:
    def __init__(self, state_dim, action_dim, max_action, args, log_out_dir):

        self.actor = Actor(state_dim, action_dim, max_action, log_out_dir).to(device)  # 动作 网络
        self.actor_target = Actor(state_dim, action_dim, max_action, log_out_dir).to(device)  # target 动作网络
        self.critic_1 = Critic(state_dim, action_dim).to(device)  #
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        # 可视化 Actor 和 Critic 网络
        self.actor.eval()
        self.critic_1.eval()
        actor_graph = hl.build_graph(self.actor, torch.zeros([1, args.state_dim]).to(device))
        critic_graph = hl.build_graph(self.critic_1, (torch.zeros([1,  args.state_dim]).to(device), torch.zeros([1, args.msg_biggest_num - args.msg_smallest_num + 1]).to(device)))
        actor_graph.theme = hl.graph.THEMES["blue"].copy()
        critic_graph.theme = hl.graph.THEMES["blue"].copy()
        actor_graph.save(f"{log_out_dir}/actor.png", format='png')
        critic_graph.save(f"{log_out_dir}/critic.png", format='png')

        self.writer = SummaryWriter(log_out_dir)

        # self.writer.add_graph(self.actor, torch.zeros([1, 1, args.state_dim]))
        # self.writer.add_graph(self.critic_1, [torch.zeros([1,  args.state_dim]), torch.zeros([1, args.msg_biggest_num - args.msg_smallest_num + 1])])


        self.actor_optimizer = optim.Adam(self.actor.parameters(),  lr=args.reforce_lr, weight_decay=args.weight_decay)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),  lr=args.reforce_lr, weight_decay=args.weight_decay)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),  lr=args.reforce_lr, weight_decay=args.weight_decay)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        
        self.args = args

    def select_action(self, state, p=False):
        # state = torch.tensor(state.reshape(1, -1)).float().to(device)
        state = state.reshape(1, -1).clone().detach().requires_grad_(True)
        return self.actor(state, p).cpu().data.numpy().flatten()

    def update(self, num_iteration):  # 在经验里随机取 20 次

        # 首先把所有模型置为训练模式
        self.actor.train()
        self.actor_target.train()
        self.critic_1.train()
        self.critic_1_target.train()
        self.critic_2.train()
        self.critic_2_target.train()

        # if self.num_training % 500 == 0:
        # print("====================================")
        # print("model has been trained for {} times...".format(self.num_training))
        # print("====================================")
        loss_q1_list = []
        loss_q2_list = []
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(self.args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, self.args.policy_noise).to(device)  # 加入噪声用到的步骤
            noise = noise.clamp(-self.args.noise_clip, self.args.noise_clip)  # 加入噪声 用到的步骤
            next_action = (self.actor_target(next_state) + noise)  # 输出噪声

            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)  # 打分1
            target_Q2 = self.critic_2_target(next_state, next_action)  # 打分2
            target_Q = torch.min(target_Q1, target_Q2)  # 防止高估 取小
            target_Q = reward + ((1 - done) * self.args.gamma * target_Q).detach()  # 更新部分真实 reward(过程累计到现在得到的分数) 计算得到的

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            loss_q1_list.append(loss_Q1.item())
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            loss_q2_list.append(loss_Q2.item())
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)

            # Delayed policy updates:
            # 延迟更新 策略网络
            if i % self.args.policy_delay == 0:
                # Compute actor loss:
                # actor_loss = (self.critic_1(state, self.actor(state)) * torch.log(self.actor(state))).mean()
                actor_loss = - (self.critic_1(state, self.actor(state))).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1
        # 返回此次训练平均的 loss_Q1 loss_Q2
        # logger.info(f'loss_q1_list: {loss_q1_list}')
        # logger.info(f'loss_q2_list: {loss_q2_list}')
        # 首先把所有模型置为训练模式
        self.actor.eval()
        self.actor_target.eval()
        self.critic_1.eval()
        self.critic_1_target.eval()
        self.critic_2.eval()
        self.critic_2_target.eval()
        return self.num_training, sum(loss_q1_list)/len(loss_q1_list), sum(loss_q2_list)/len(loss_q2_list)

    def save(self, epcoh, val_acc, save_dir):
        import time
        time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        torch.save(self.actor.state_dict(), save_dir+'epoch_'+str(epcoh)+"_"+val_acc+'_'+time_mark+'_actor.pth')
        torch.save(self.actor_target.state_dict(), save_dir+'epoch_'+str(epcoh)+"_"+val_acc+'_'+time_mark+'_actor_target.pth')
        torch.save(self.critic_1.state_dict(), save_dir+'epoch_'+str(epcoh)+"_"+val_acc+'_'+time_mark+'_critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), save_dir+'epoch_'+str(epcoh)+"_"+val_acc+'_'+time_mark+'_critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), save_dir+'epoch_'+str(epcoh)+"_"+val_acc+'_'+time_mark+'_critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), save_dir+'epoch_'+str(epcoh)+"_"+val_acc+'_'+time_mark+'_critic_2_target.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")
        logger.info("Model has been saved...")

    def load(self):
        logger.info(f'模型路径为: {self.args.model_load_dir}_')
        self.actor.load_state_dict(torch.load(self.args.model_load_dir + '_actor.pth'))
        self.actor_target.load_state_dict(torch.load(self.args.model_load_dir + '_actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(self.args.model_load_dir + '_critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(self.args.model_load_dir + '_critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(self.args.model_load_dir + '_critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(self.args.model_load_dir + '_critic_2_target.pth'))
        # print("====================================")
        # print("model has been loaded...")
        # print("====================================")
        logger.info("model has been loaded...")

