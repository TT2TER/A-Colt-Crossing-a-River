import torch
import torch.nn as nn

L = 6  # 序列长度
N = 5  # 批量大小
d = 4  # 嵌入维度
h = 3  # 隐层大小
n = 2  # LSTM的深度
lstm = nn.LSTM(d, h, num_layers=n, bidirectional=True)

inputs = torch.randn(L, N, d)  # LSTM的输入
output, (h_n, c_n) = lstm(inputs)  # LSTM的输出

output = output.reshape(L, N, 2, h)
forward_output = output[:, :, 0, :]  # 正向LSTM的输出，形状为(L, N, h)
backward_output = output[:, :, 1, :]  # 反向LSTM的输出，形状为(L, N, h)
forward_h_n = h_n[::2]  # 正向LSTM的h_n，形状为(n, N, h)
backward_h_n = h_n[1::2]  # 反向LSTM的h_n，形状为(n, N, h)

# 因为是正向LSTM，所以时间方向是从左向右，因此forward_output[-1]代表
# 最后一个时间步上的最后一层的输出
print(forward_output[-1] == forward_h_n[-1])

# 因为是反向LSTM，所以时间方向是从右向左，因此backward_output[0]代表
# 最后一个时间步上的最后一层的输出
print(backward_output[0] == backward_h_n[-1])
