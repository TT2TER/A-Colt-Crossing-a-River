import numpy as np

def egreedy_policy(q_values, state, epsilon=0.1):
    """
    输入参数：
    q_values：一个包含了每个状态-动作对的Q值的数组（Q-table）。
    state：当前智能体所处的状态。

    返回值：
        选择的动作的索引，表示智能体将要执行的动作。

    参数 epsilon（ε）是一个控制随机性的参数，通常取小于1的小数值。函数的行为如下：
        如果生成的随机数小于 epsilon，函数会以概率 epsilon 选择一个随机动作（探索）。
        如果生成的随机数大于等于 epsilon，函数会选择具有最高Q值的动作（利用）。
    """
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(q_values[state])