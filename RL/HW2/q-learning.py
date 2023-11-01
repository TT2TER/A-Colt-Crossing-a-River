from GRIDWORLD import GridWorld
from EGREEDY import egreedy_policy
from SHOW import play
import numpy as np
import matplotlib.pyplot as plt

def q_learning(
    env,                   # 环境对象，表示智能体的操作环境
    num_episodes=500,      # 训练的总周期数（默认为500个周期）
    render=True,           # 是否在每个周期中渲染环境（默认为True）
    exploration_rate=0.1,  # 探索率（ε-greedy策略中的ε值，默认为0.1）
    learning_rate=0.5,     # 学习率（默认为0.5）
    gamma=0.9,             # 折扣因子（默认为0.9）
):
    # 初始化Q值表，所有Q值都初始化为0
    q_values = np.zeros((num_states, num_actions))
    
    # 存储每个周期的总奖励
    ep_rewards = []

    # 开始训练循环，每个周期都是一个episode
    for _ in range(num_episodes):
        # 重置环境，开始新的episode
        state = env.reset()
        done = False
        reward_sum = 0

        # 在一个episode中不断选择动作并与环境交互，直到episode结束
        while not done:
            # 使用ε-greedy策略选择动作
            action = egreedy_policy(q_values, state, exploration_rate)
            
            # 执行选定的动作并观察环境的反馈
            next_state, reward, done = env.step(action)
            
            # 累积奖励
            reward_sum += reward

            # 更新Q值
            # 根据Q-learning的更新规则计算目标Q值 (td_target)
            td_target = reward + gamma * np.max(q_values[next_state])
            # 计算TD误差
            td_error = td_target - q_values[state][action]
            # 使用学习率更新Q值
            q_values[state][action] += learning_rate * td_error
            
            # 更新状态为下一个状态
            state = next_state

            # 如果需要，渲染当前环境状态
            if render:
                env.render(q_values, action=actions[action], colorize_q=True)

        # 存储本周期的总奖励
        ep_rewards.append(reward_sum)

    # 返回每个周期的总奖励列表和学习后的Q值表
    return ep_rewards, q_values



if __name__ == "__main__":
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

    env = GridWorld()
    num_states = 4 * 12
    num_actions = 4

    q_learning_rewards, q_values = q_learning(
        env, gamma=0.9, learning_rate=1, render=False
    )
    env.render(q_values, colorize_q=True)

    q_learning_rewards, _ = zip(
        *[
            q_learning(env, render=False, exploration_rate=0.1, learning_rate=1)
            for _ in range(10)
        ]
    )
    avg_rewards = np.mean(q_learning_rewards, axis=0)
    mean_reward = [np.mean(avg_rewards)] * len(avg_rewards)

    fig, ax = plt.subplots()
    ax.set_xlabel("Episodes using Q-learning")
    ax.set_ylabel("Rewards")
    ax.plot(avg_rewards)
    ax.plot(mean_reward, "g--")

    # 平滑的奖励曲线
    window_size = 100
    smooth_avg_rewards = (
        np.convolve(avg_rewards, np.ones(window_size), "valid") / window_size
    )
    smooth_mean_reward = [np.mean(smooth_avg_rewards)] * len(smooth_avg_rewards)
    fig, ax = plt.subplots()
    ax.set_xlabel("Episodes using Q-learning")
    ax.set_ylabel("Rewards")
    ax.plot(smooth_avg_rewards)
    ax.plot(smooth_mean_reward, "g--")

    print("Mean Reward using Q-Learning: {}".format(mean_reward[0]))

    play(q_values)
