import os
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
import time
from torchviz import make_dot
from Replay import ReplayBuffer
import math

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, out_dim * atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
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
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
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
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, atom_size, self.support
        ).to(self.device)
        #随机初始化dqn参数
        self.dqn.apply(self._weights_init)
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

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)#每个原子的间隔

        with torch.no_grad():#不计算梯度
            next_action = self.dqn_target(next_state).argmax(1)#选择最大的动作
            next_dist = self.dqn_target.dist(next_state)#得到下一个状态的分布
            next_dist = next_dist[range(self.batch_size), next_action]#得到下一个状态的分布的最大动作的分布

            t_z = reward + (1 - done) * self.gamma * self.support#计算目标分布
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)#限制目标分布的范围
            b = (t_z - self.v_min) / delta_z#计算目标分布的原子位置
            l = b.floor().long()#向下取整
            u = b.ceil().long()#向上取整

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()#计算偏移量
                .unsqueeze(1)#增加维度
                .expand(self.batch_size, self.atom_size)#扩展维度
                .to(self.device)
            )#offset的维度为batch_size*atom_size

            proj_dist = torch.zeros(next_dist.size(), device=self.device)#初始化proj_dist
            proj_dist.view(-1).index_add_(#index_add_()函数的作用是将一个张量tensor中的某些位置的元素累加到另一张量中，其参数为：dim：指定的维度；index：指定的位置；source：源张量
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)#得到当前状态的分布
        log_p = torch.log(dist[range(self.batch_size), action])#得到当前状态的分布的动作的概率的对数

        loss = -(proj_dist * log_p).sum(1).mean()#


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