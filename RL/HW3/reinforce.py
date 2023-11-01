from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import time
import torch
torch.manual_seed(0)  # set random seed


class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)  # 全连接层，将输入状态映射到隐藏层
        # self.fc2 = nn.Linear(h_size, h_size)  # 第二个隐藏层，没用到
        self.fc3 = nn.Linear(h_size, a_size)  # 将隐藏层的输出映射到输出层

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)  # 对分数进行 Softmax 操作，得到动作的概率分布

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(
            0).to(device)  # 将环境传递的状态转换为 PyTorch 张量
        probs = self.forward(state).cpu()  # 使用策略网络的前向传播计算动作的概率分布
        # Categorical 分布是离散概率分布的一种，用于描述随机变量的可能取值集合是有限的情况。
        m = Categorical(probs)
        # 在强化学习中，通常用来表示在一个离散动作空间中选择动作的概率分布。
        action = m.sample()  # 从概率分布中采样一个动作（考虑每个动作的概率）
        # 返回选定的动作及其动作的对数概率，对数概率将在策略梯度算法中用于梯度更新。
        return action.item(), m.log_prob(action)

    def pre(self, state):
        state = torch.from_numpy(state).float().unsqueeze(
            0).to(device)  # 将环境传递的状态转换为 PyTorch 张量
        probs = self.forward(state).cpu()  # 使用策略网络的前向传播计算动作的概率分布
        # Categorical 分布是离散概率分布的一种，用于描述随机变量的可能取值集合是有限的情况。
        m = Categorical(probs)
        # 选择概率最高的一个动作
        action = torch.argmax(probs)
        return action.item(), m.log_prob(action)


def reinforce(n_episode=10000, max_t=1000, gamma=0.99, print_every=10):
    # n_episode: 训练的 episode 数量
    # max_t: 每个 episode 进行的最大步数
    # gamma: 折扣因子
    # print_every: 每训练 print_every 个 episode 打印一次信息
    scores_deque = deque(maxlen=100)  # 用双端队列保存最近100次的得分
    scores = []  # 保存每次得分
    for i_episode in range(1, n_episode+1):
        saved_log_porbs = []  # 存储每个步的动作对数概率
        rewards = []  # 储每个步的奖励
        state = env.reset()  # 初始化环境
        for t in range(max_t):
            action, log_prob = policy.act(state)  # 根据当前策略 policy 选择动作
            saved_log_porbs.append(log_prob)  # 将该步的对数概率保存，用来计算策略梯度
            state, reward, done, _ = env.step(
                action)  # 执行该步动作，得到下一步的状态、奖励(这一个动作活着就是正奖励）、是否结束
            rewards.append(reward)
            if done:
                break  # gemeover一次采样结束

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # discounts = [gamma**i for i in range(len(rewards)+1)]#计算折扣因子数组 discounts
        # R = sum([a*b for a, b in zip(discounts, rewards)])# 将 discounts 和 rewards 列表逐一组合成一个元组的迭代器，也就是在每个时间步上，它会生成一个包含折扣因子和奖励的元组。例如，如果 discounts 为 [0.1, 0.01, 0.001]，rewards 为 [1, 2, 3]，那么 zip 会生成迭代器 (0.1, 1), (0.01, 2), (0.001, 3)
        # policy_loss = []
        # for log_prob in saved_log_porbs:
        #     policy_loss.append(-log_prob * R)#这个对数概率在计算的时候已经考虑了独热向量，所以这里不需要再乘以独热向量，直接乘以 R 即可
        # policy_loss = torch.cat(policy_loss)
        # policy_loss = policy_loss.sum()#计算策略损失
        # optimizer.zero_grad()#清零优化器梯度缓冲区，PyTorch 默认情况下会保留之前的梯度信息
        # policy_loss.backward()#自动计算策略损失相对于网络参数的梯度
        # optimizer.step()#更新网络的参数

        G = 0
        optimizer.zero_grad()
        for i in reversed(range(len(rewards))):
            # d= np.mean(rewards[:])
            reward = rewards[i]
            log_prob = saved_log_porbs[i]
            G = gamma * G + reward  # - d
            loss = -log_prob * G
            loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\t Average Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 498.0:
            print('Environment solved in {:d} episode!\tAverage Score:{:.2f}'.format(
                i_episode-100, np.mean(scores_deque)))
            break

    return scores

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 控制是否允许多个线程同时加载和使用 MKL 库
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # 调用GPU
    env = gym.make('CartPole-v1')  # 创建环境
    seed = 520
    env.seed(20020917)
    np.random.seed(seed)
    torch.manual_seed(0)  # cpu
    torch.cuda.manual_seed(0)  # GPU
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
    # print('observation space:', env.observation_space)
    # print('action space:', env.action_space)
    policy = Policy().to(device)
    # policy.parameters()返回策略网络 policy 中的所有可学习参数 ；lr 学习率
    optimizer = optim.Adam(policy.parameters(), lr=1e-2, betas=[0.9, 0.999])
    scores = reinforce()  # 使用reinforce算法进行训练

    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    # 将结果平滑输出
    window = 50
    smoothed_scores = [np.mean(scores[i-window:i+1])
                    for i in range(window, len(scores))]
    ax = fig.add_subplot(122)
    plt.plot(np.arange(1, len(smoothed_scores)+1), smoothed_scores)
    plt.ylabel('Smoothed reward')
    plt.xlabel('Episode')
    plt.show()

    # 可视化最终策略
    env = gym.make('CartPole-v1', render_mode='human')

    state = env.reset()
    for t in range(1000):
        action, _ = policy.pre(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            print("reward:", t+1)
            break

    env.close()
