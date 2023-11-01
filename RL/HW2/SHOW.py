from GRIDWORLD import GridWorld
from EGREEDY import egreedy_policy
import numpy as np

actions = ["UP", "DOWN", "RIGHT", "LEFT"]

def play(q_values):
    # 根据最终训练的Q值结果，显示选择动作的过程
    env = GridWorld()
    state = env.reset()
    done = False

    while not done:
        # 每次选择相应动作后都可视化一次表格
        # 选择动作
        action = egreedy_policy(q_values, state, 0.0)
        # 执行动作
        next_state, reward, done = env.step(action)
        # 更新状态
        state = next_state
        # print(state)
        # 画图
        env.render(q_values=q_values, action=actions[action], colorize_q=True)