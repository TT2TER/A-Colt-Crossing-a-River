import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import torch
from torch.distributions import Categorical
import copy
import gym
import os
import numpy as np
import matplotlib.pyplot as plt
import time


class Actor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.fc3(x), dim=1)
        return probs


class ReplayBuffer:
    '''每次采样所有样本'''

    def __init__(self):
        self.buffer = deque()

    def push(self, transition):
        self.buffer.append(transition)  # 将transition加入buffer中

    def clear(self):
        self.buffer.clear()

    def sample(self):
        batch = list(self.buffer)
        return zip(*batch)  # 用于将一个序列（例如列表或元组）中的元素解包为独立的参数


class PPO:
    def __init__(self, n_states, n_actions, update_freq, K_epochs, 
                 gamma=0.99, device='gpu', hidden_dim=256, lr=3e-4, eps_clip=0.2):
        self.gamma = gamma  # 折扣因子
        self.device = device  # 计算设备
        self.actor = Actor(n_states, hidden_dim, n_actions).to(self.device)  # 创建一个 Actor 模型，用于学习策略。
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=[0.9,0.999])  # 创建 Actor 模型的优化器。
        self.memory = ReplayBuffer()  # 创建一个经验回放池，用于存储智能体与环境交互得到的经验数据。
        self.K_epochs = K_epochs  # 设置 PPO 算法中的策略更新的迭代次数k
        self.eps_clip = eps_clip  # 近端策略优化裁剪
        self.sample_count = 0  # 用于跟踪智能体与环境交互的次数。
        self.update_freq = update_freq  # 多少次交互后，智能体会执行一次策略更新。

    def sample_action(self,state):
        #采样动作
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)# 将环境传递的状态转换为 PyTorch 张量
        probs = self.actor(state) #通过 Actor 模型计算给定状态下的动作概率分布。
        dist = Categorical(probs) #使用概率分布创建一个 Categorical 分布对象。
        action = dist.sample() #从动作概率分布中采样一个动作。
        log_probs = dist.log_prob(action).detach() #计算采样动作的对数概率，并将其存储在 log_probs 中，同时使用 detach 方法将其与actor的计算图分离，避免反向传播时对actor的参数进行更新。
        return action.detach().cpu().numpy().item(), log_probs# 返回采样的动作
    
    @torch.no_grad()#该装饰器表示以下函数不需要计算梯度
    def test_action(self,state):
        #用于预测动作
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()# 返回采样的动作
    
    @torch.no_grad()#该装饰器表示以下函数不需要计算梯度
    def pre_action(self,state):
        #用于展示概率最高动作
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        action = torch.argmax(probs)
        return action.detach().cpu().numpy().item()# 返回采样的动作
    
    def update(self):
        #执行策略更新的方法，在update_freq次后执行
        if self.sample_count % self.update_freq != 0:
            return# 没到update_freq次时不执行
        
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()#经验回放池中采样旧的状态、动作、对数概率、奖励和终止状态信息

        #转换数据为张量形式
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)

        # 蒙特卡洛估计每个状态的回报
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        
        # 将回报归一化 在理论上不是必须的，但在实践中，它减少了回报的方差，使收敛更加稳定和快速。
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero

        for _ in range(self.K_epochs):
            #迭代k次更新
            # 使用PPO-clip
            values = returns.mean() #计算旧状态的价值作为基线
            advantage = returns - values
            # get action probabilities
            probs = self.actor(old_states)# 使用 Actor 模型计算旧状态的动作概率
            dist = Categorical(probs) #创建一个 Categorical 分布对象，用于计算动作的概率
            
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs) # 计算策略比例（pi_theta / pi_theta_old），这是新策略和旧策略的比值。old_log_probs 必须是已分离的，以避免梯度传播到旧策略。
            
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
        self.memory.clear()


def init():
    env = gym.make('CartPole-v1')

    # 初始化随机数
    seed = 20020917
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # GPU
    # 设置 Python 的哈希种子。
    # 在某些情况下，Python 的哈希值可以影响一些数据结构的顺序，通过设置这个种子，可以确保在不同运行中哈希值的计算是一致的。
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 控制 PyTorch 中的 cuDNN 库的行为
    # 此选项确保 cuDNN 使用确定性算法执行操作。这意味着相同的操作在不同运行中将生成相同的结果。
    # 这对于实现训练的可重现性非常重要，因为在深度学习中，有些优化可能涉及到随机性操作，例如权重初始化
    torch.backends.cudnn.deterministic = True
    # cuDNN 尝试自动寻找最适合你的硬件的实现以获得最佳性能。它会根据硬件性能特征选择不同的算法，以获得最佳的速度。
    # 如果你的输入大小或类型在运行时改变，那么它可能会选择不同的算法，这会导致性能波动。
    torch.backends.cudnn.benchmark = False
    # 禁用 cuDNN 可能会降低性能，但有时它对于特定任务和硬件配置是必要的，特别是在需要更高可重现性的情况下。
    torch.backends.cudnn.enabled = False
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # 调用GPU

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPO(n_states, n_actions, update_freq=200, K_epochs=40, gamma=0.99, device=device, hidden_dim=16, lr=5e-3, eps_clip=0.2)

    return env, agent, device

def train(env,agent):
    rewards = []#记录所有回合的奖励
    best_reward = 0.0#记录最佳回合的奖励
    output_agent = None
    train_episodes = 1000#训练回合数
    max_steps = 1000#每回合的最大训练步数
    eval_freq = 5#每多少回合进行一次评估
    eval_episodes = 20#每次评估的回合数
    for i_episode in range(train_episodes):
        ep_reward = 0.0 #记录每个回合的奖励
        ep_steps = 0
        state=env.reset() #初始化环境
        for _ in range(max_steps):
            ep_steps += 1
            action, log_probs = agent.sample_action(state) #采样动作
            next_state, reward, done, _ = env.step(action) #执行动作，更新环境
            agent.memory.push([state, action, log_probs, reward, done]) #保存transition
            state = next_state #state更新到下一个状态
            agent.update() #更新智能体(每update_freq次采样后更新)
            ep_reward += reward #累加回合奖励
            if done:
                break #回合结束，跳出循环

        if (i_episode+1)% eval_freq == 0:
            #每eval_freq个回合进行一次评估，记录最佳智能体
            sum_eval_reward = 0.0
            for _ in range (eval_episodes):
                eval_ep_reward = 0.0
                state=env.reset()
                for _ in range(max_steps):
                    action = agent.test_action(state) #选择动作
                    next_state, reward, done, _ = env.step(action) #执行动作，更新环境
                    state = next_state #state更新到下一个状态
                    eval_ep_reward += reward #累加回合奖励
                    if done:
                        break
                sum_eval_reward += eval_ep_reward #将回合奖励累加到评估奖励中
            mean_eval_reward = sum_eval_reward / eval_episodes
            if mean_eval_reward > best_reward:
                best_reward = mean_eval_reward
                output_agent = copy.deepcopy(agent)
                print(f"episode:{i_episode+1}/{train_episodes}，rewards:{ep_reward:.2f}，eval_rewards:{mean_eval_reward:.2f}，our_best_rewards:{best_reward:.2f}，updated!")
            else:
                print(f"episode:{i_episode+1}/{train_episodes}，rewards:{ep_reward:.2f}，eval_rewards:{mean_eval_reward:.2f}，our_best_rewards:{best_reward:.2f}，don't need to update.")
            if mean_eval_reward >= 498.0:
                print('Environment solved in {:d} episode!\tAverage Score:{:.2f}'.format(i_episode, mean_eval_reward))
                break
        rewards.append(ep_reward)
    print(f"train over. best_reward:{best_reward:.2f}")
    env.close()
    return output_agent, {"rewards":rewards}

def test(env, agent):
    rewards = []  # 记录所有回合的奖励
    test_episodes = 10  # 测试回合数
    max_steps = 1000#每回合的最大步数
    for i_episode in range(test_episodes):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(max_steps):
            ep_step+=1
            action = agent.test_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        rewards.append(ep_reward)
        print(f"episode:{i_episode+1}/{test_episodes}，rewards:{ep_reward:.2f}")
    print("test over.")
    env.close()
    return {'rewards':rewards}


def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_rewards(rewards, tag='train', device='cpu'):
    ''' 画图
    '''
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {device} of ppo for cartpole")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.show()

if __name__ == "__main__":
    env, agent, device = init()
    #训练
    best_agent, res_diction = train(env, agent)

    #测试
    res_dic = test(env, best_agent)

    #画图
    plot_rewards(res_diction['rewards'],tag="train",device=device)
    plot_rewards(res_dic['rewards'],tag="test",device=device)

    # 可视化最终策略
    env = gym.make('CartPole-v1', render_mode='human')

    state = env.reset()
    for t in range(1000):
        action = best_agent.pre_action(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            print("reward:", t+1)
            break
    env.close()
