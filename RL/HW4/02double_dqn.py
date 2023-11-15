from typing import Dict, List, Tuple
from Replay import ReplayBuffer
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
import time



class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 32), 
            nn.ReLU(),
            nn.Linear(32, 32), 
            nn.ReLU(), 
            nn.Linear(32, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        @torch.no_grad()
        def _weights_init(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

        self._weights_init = _weights_init

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        #随机初始化dqn参数
        self.dqn.apply(self._weights_init)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())# 将dqn的参数复制给dqn_target
        self.dqn_target.eval() # eval()：这一部分将 dqn_target 设置为评估模式，即在训练过程中不会计算梯度。这是因为在训练过程中，我们只需要使用 dqn_target 来计算目标值，而不需要更新它的参数。
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(),lr = 1e-3)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state.
        用epsilon greedy策略选择动作
        将动作和状态暂存入transition（如果是训练）
        返回选择的动作
        """
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax() 
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env.
        执行action
        如果是训练，存储transition
        将transition存到memory中
        """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated #终止或被截断

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)#存储transition
            '''星号（*）操作符用于解包（unpack）序列。在这个上下文中，*self.transition将self.transition这个序列中的所有元素解包，然后传递给self.memory.store方法。

假设self.transition是一个包含五个元素的元组（或列表），例如self.transition = (obs, act, rew, next_obs, done)，那么*self.transition就相当于obs, act, rew, next_obs, done'''
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()#从memory中随机采样,返回的是字典，值是batch_size大小的数组
        # print(samples)
        # time.sleep(500)
        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)#选择动作
            next_state, reward, done = self.step(action)#执行动作，并存储结果

            state = next_state
            score += reward

            # if episode ends
            if done:
                state, _ = self.env.reset(seed=self.seed)#初始化环境
                scores.append(score)#记录结果
                score = 0

            # if training is ready
            # batch_size多少词数据训练一回model
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)# 记录loss
                update_cnt += 1
                
                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()# 更新target网络

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)
                
        self.env.close()
                
    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True
        
        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal(这个状态不是终止状态)
        #       = r                       otherwise（终止在了这个状态）
        # print("state",state)
        # print("action",action)
        curr_q_value = self.dqn(state)
        # print("curr_q_value",curr_q_value)
        curr_q_value=curr_q_value.gather(1, action)
        # print("curr_q_value",curr_q_value)
        # time.sleep(500)
        '''.gather(1, action)：这一部分使用 .gather 方法来选择张量中的特定元素。在这里，1 表示选择的维度，通常是动作的索引，而 action 是一个张量，包含要选择的动作的索引。

具体来说，.gather 操作会根据 action 中的索引值，从 self.dqn(state) 返回的张量中选择对应索引位置的值。这是在Q-learning等强化学习算法中常用的操作，用于计算当前状态下选择的动作的Q值。'''
        
        '''
        ddqn 使用dqn(随时更新网络进行选择)
        使用target进行计算
        '''
        selected_action = self.dqn(next_state).argmax(dim=1, keepdim=True)
        next_q_value = self.dqn_target(next_state).gather(  # Double DQN
            1, selected_action
        ).detach()
        #print("next_q_value",next_q_value)
        #time.sleep(500)
        '''self.dqn_target(next_state)：这一部分是将状态 next_state 传递给神经网络模型 self.dqn_target，
        并得到模型的输出。这通常是一个Q值估计的张量，其中每行表示不同的动作。

        .max(dim=1, keepdim=True)：这一部分使用 .max 方法在每行上进行操作，dim=1 表示在每行上寻找最大值。
        keepdim=True 的设置保持了结果的维度一致。

        [0]：这一部分用于提取 .max 操作的结果中的最大值。.max 操作返回一个元组，第一个元素是最大值的张量，
        第二个元素是最大值的索引。

        .detach()：这一部分用于分离张量，即从计算图中分离，以避免梯度传播。通常在需要计算梯度的张量和不需要计算梯度的张量之间切换时使用。'''
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)
        # print("target",target)
        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)
        # print("loss",loss)
        # 生成计算图
        # dot = make_dot(loss)

        # # 显示计算图
        # dot.view()
        # time.sleep(500)
        '''
        Smooth L1损失是一种在回归问题中常用的损失函数，它是平方损失和L1损失的结合。当预测值和真实值之间的差距很小（小于1）时，
        Smooth L1损失表现为平方损失，当差距较大时，表现为L1损失。这样做的好处是，Smooth L1损失在处理离群点（outliers）时比平方损失更稳定，
        而在差距较小的情况下，又能保持平方损失的良好性质。
        '''

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
            #画b图
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()

# environment
env = gym.make("CartPole-v1", max_episode_steps=500, render_mode="rgb_array")

seed = 777

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)

# parameters
num_frames = 10000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 1 / 2000

if __name__ == "__main__":
    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, seed)
    agent.train(num_frames,5000)