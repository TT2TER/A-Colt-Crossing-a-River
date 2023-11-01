from GRIDWORLD import GridWorld
from EGREEDY import egreedy_policy
from SHOW import play
import numpy as np
import matplotlib.pyplot as plt

def sarsa(
    env,
    num_episodes=500,
    render=True,
    exploration_rate=0.1,
    learning_rate=0.5,
    gamma=0.9,
):
    # 初始化
    q_values_sarsa = np.zeros((num_states, num_actions))
    ep_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False  # 回合结束标志
        reward_sum = 0

        # 根据epsilon greedy policy选择动作
        action = egreedy_policy(q_values_sarsa, state, exploration_rate)

        while not done:
            # 执行动作
            next_state, reward, done = env.step(action)
            reward_sum += reward

            # 下个状态下选择对应动作
            next_action = egreedy_policy(q_values_sarsa, next_state, exploration_rate)
            # Next q value is the value of the next action
            # 计算时间差分的目标、误差，根据计算结果更新q值
            td_target = reward + gamma * q_values_sarsa[next_state][next_action]
            td_error = td_target - q_values_sarsa[state][action]
            q_values_sarsa[state][action] += learning_rate * td_error

            # 更新 state、action
            state = next_state
            action = next_action

            if render:
                env.render(q_values_sarsa, action=actions[action], colorize_q=True)

        ep_rewards.append(reward_sum)

    return ep_rewards, q_values_sarsa


if __name__ == "__main__":
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

    env = GridWorld()
    num_states = 4 * 12
    num_actions = 4


    sarsa_rewards, q_values_sarsa = sarsa(env, render=False, learning_rate=0.5, gamma=0.99)


    sarsa_rewards, _ = zip(
        *[sarsa(env, render=False, exploration_rate=0.2) for _ in range(10)]
    )
    avg_rewards = np.mean(sarsa_rewards, axis=0)
    mean_reward = [np.mean(avg_rewards)] * len(avg_rewards)

    fig, ax = plt.subplots()
    ax.set_xlabel("Episodes using Sarsa")
    ax.set_ylabel("Rewards")
    ax.plot(avg_rewards)
    ax.plot(mean_reward, "g--")

    print("Mean Reward using Sarsa: {}".format(mean_reward[0]))

    play(q_values_sarsa)
