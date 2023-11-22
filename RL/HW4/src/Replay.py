import numpy as np
from typing import Dict,List,Tuple,Deque
from collections import deque
from segment_tree import MinSegmentTree, SumSegmentTree
import random

# class ReplayBuffer:
#     """A simple numpy replay buffer."""

#     def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
#         self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         '''在深度学习和机器学习中，32位浮点数（float32）常常被用作默认的数据类型，原因有两个：

# 精度：对于大多数应用来说，32位浮点数提供的精度已经足够。它能够表示大约7位十进制数，这对于大多数机器学习任务来说已经足够。

# 计算效率：相比64位浮点数（float64），32位浮点数在计算上更加高效。这是因为32位浮点数占用的内存更少，可以减少内存带宽和缓存的使用，从而提高计算效率。'''
#         self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.acts_buf = np.zeros([size], dtype=np.float32)
#         self.rews_buf = np.zeros([size], dtype=np.float32)
#         self.done_buf = np.zeros([size], dtype=np.float32)
#         self.max_size, self.batch_size = size, batch_size
#         self.ptr, self.size, = 0, 0

#     def store(
#         self,
#         obs: np.ndarray,
#         act: np.ndarray, 
#         rew: float, 
#         next_obs: np.ndarray, 
#         done: bool,
#     ):
#         self.obs_buf[self.ptr] = obs
#         self.next_obs_buf[self.ptr] = next_obs
#         self.acts_buf[self.ptr] = act
#         self.rews_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)

#     def sample_batch(self) -> Dict[str, np.ndarray]:
#         idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
#         return dict(obs=self.obs_buf[idxs],
#                     next_obs=self.next_obs_buf[idxs],
#                     acts=self.acts_buf[idxs],
#                     rews=self.rews_buf[idxs],
#                     done=self.done_buf[idxs])

#     def __len__(self) -> int:
#         return self.size


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32, n_step: int = 3, gamma: float = 0.99,):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        '''在深度学习和机器学习中，32位浮点数（float32）常常被用作默认的数据类型，原因有两个：

精度：对于大多数应用来说，32位浮点数提供的精度已经足够。它能够表示大约7位十进制数，这对于大多数机器学习任务来说已经足够。

计算效率：相比64位浮点数（float64），32位浮点数在计算上更加高效。这是因为32位浮点数占用的内存更少，可以减少内存带宽和缓存的使用，从而提高计算效率。'''
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(
            self.size, size=self.batch_size, replace=False
        )

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            # for N-step Learning
            indices=indices,
        )
    
    def sample_batch_from_idxs(
        self, indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size
    


# class PrioritizedReplayBuffer(ReplayBuffer):
#     """Prioritized Replay buffer.
    
#     Attributes:
#         max_priority (float): max priority
#         tree_ptr (int): next index of tree
#         alpha (float): alpha parameter for prioritized replay buffer
#         sum_tree (SumSegmentTree): sum tree for prior
#         min_tree (MinSegmentTree): min tree for min prior to get max weight
        
#     """
    
#     def __init__(
#         self, 
#         obs_dim: int,
#         size: int, 
#         batch_size: int = 32, 
#         alpha: float = 0.6
#     ):
#         """Initialization."""
#         assert alpha >= 0
        
#         super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
#         self.max_priority, self.tree_ptr = 1.0, 0
#         self.alpha = alpha
        
#         # capacity must be positive and a power of 2.
#         tree_capacity = 1
#         while tree_capacity < self.max_size:
#             tree_capacity *= 2
#         '''
# 1. `SumSegmentTree`类：
#    - `__init__`方法：初始化函数，调用父类的初始化函数，并设置操作为加法，初始值为0.0。
#    - `sum`方法：返回指定范围内的元素之和。
#    - `retrieve`方法：找到树中第一个大于upperbound的元素的索引。

# 2. `MinSegmentTree`类：
#    - `__init__`方法：初始化函数，调用父类的初始化函数，并设置操作为取最小值，初始值为正无穷。
#    - `min`方法：返回指定范围内的最小元素。'''
#         self.sum_tree = SumSegmentTree(tree_capacity)
#         self.min_tree = MinSegmentTree(tree_capacity)
        
#     def store(
#         self, 
#         obs: np.ndarray, 
#         act: int, 
#         rew: float, 
#         next_obs: np.ndarray, 
#         done: bool
#     ):
#         """Store experience and priority."""
#         super().store(obs, act, rew, next_obs, done)
#         #调用父类的store方法，存储经验
        
#         self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
#         #将当前经验的优先级存储到sum_tree中，优先级的计算公式为：优先级 = 最大优先级 ** alpha
#         self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
#         #将当前经验的优先级存储到min_tree中，优先级的计算公式为：优先级 = 最大优先级 ** alpha
#         self.tree_ptr = (self.tree_ptr + 1) % self.max_size
#         #更新tree_ptr的值，如果超过最大容量，则从0重新开始

#     def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
#         """Sample a batch of experiences."""
#         assert len(self) >= self.batch_size
#         #如果经验池中的经验数量小于batch_size，则报错
#         assert beta > 0
#         #如果beta小于等于0，则报错

        
#         indices = self._sample_proportional()#得到基于比例的索引表
        
#         obs = self.obs_buf[indices]
#         next_obs = self.next_obs_buf[indices]
#         acts = self.acts_buf[indices]
#         rews = self.rews_buf[indices]
#         done = self.done_buf[indices]
#         weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
#         return dict(
#             obs=obs,
#             next_obs=next_obs,
#             acts=acts,
#             rews=rews,
#             done=done,
#             weights=weights,
#             indices=indices,
#         )
        
#     def update_priorities(self, indices: List[int], priorities: np.ndarray):
#         """Update priorities of sampled transitions."""
#         # print(priorities)
#         assert len(indices) == len(priorities)#如果索引表和优先级表的长度不一致，则报错
        
#         for idx, priority in zip(indices, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self)
#             #对于每一个索引和优先级，如果优先级小于等于0，则报错；如果索引小于0或者大于等于经验池的大小（不在有效范围内），则报错

#             self.sum_tree[idx] = priority ** self.alpha#更新sum_tree中的优先级
#             self.min_tree[idx] = priority ** self.alpha#更新min_tree中的优先级

#             self.max_priority = max(self.max_priority, priority)#更新最大优先级，取当前优先级和最大优先级的最大值
            
#     def _sample_proportional(self) -> List[int]:
#         """Sample indices based on proportions."""
#         indices = []#索引表
#         p_total = self.sum_tree.sum(0, len(self) - 1)#计算优先级总和
#         segment = p_total / self.batch_size#将总和分成batch_size份
        
#         for i in range(self.batch_size):
#             #对每一个元素，计算段的上下界
#             a = segment * i
#             b = segment * (i + 1)
#             upperbound = random.uniform(a, b)#随机生成一个上界
#             idx = self.sum_tree.retrieve(upperbound)#找到第一个大于upperbound的元素的索引
#             indices.append(idx)#将索引添加到索引表中
            
#         return indices#返回索引表
    
#     def _calculate_weight(self, idx: int, beta: float):
#         """Calculate the weight of the experience at idx."""
#         # get max weight
#         p_min = self.min_tree.min() / self.sum_tree.sum()#计算最小优先级除以总优先级，得到比例
#         max_weight = (p_min * len(self)) ** (-beta)#计算最大权重
        
#         # calculate weights
#         p_sample = self.sum_tree[idx] / self.sum_tree.sum()
#         weight = (p_sample * len(self)) ** (-beta)#原始权重
#         weight = weight / max_weight#原始权重除以最大权重，得到最终权重
        
#         return weight
class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight