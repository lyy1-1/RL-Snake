import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from copy import deepcopy

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # ReLU 引入非线性，网络输出各动作的 Q 值，不是概率。
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, target_update_interval=1000, *, use_target_network=True):
        if (isinstance(target_update_interval, bool)
                or not isinstance(target_update_interval, int)
                or target_update_interval <= 0):
            raise ValueError('target_update_interval must be a positive integer')
        self.lr = lr
        self.gamma = gamma
        # 在线网络 Q_θ 用于动作选择，并通过 TD 损失学习。
        self.model = model
        # 目标网络 Q_θ⁻ 是在线网络的独立副本，提供暂时固定的参考值。
        self.target_model = deepcopy(model) if use_target_network else None
        if self.target_model is not None:
            # eval 切换评估模式，requires_grad_ 冻结目标参数。
            self.target_model.eval().requires_grad_(False)
        self.target_update_interval = target_update_interval
        self.n_updates = 0
        # 优化器只更新在线网络。
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.as_tensor(np.asarray(state), dtype=torch.float)
        next_state = torch.as_tensor(np.asarray(next_state), dtype=torch.float)
        action = torch.as_tensor(np.asarray(action), dtype=torch.long)
        reward = torch.as_tensor(np.asarray(reward), dtype=torch.float)

        if len(state.shape) == 1:
            # 统一为批量形式：单条观察 [11] → [1,11]，单条动作 [3] → [1,3]。
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 预测当前观察的 Q 值，保留计算图用于反向传播。
        pred = self.model(state)

        # detach 切断目标的梯度路径，clone 让目标与预测使用独立数据。
        # 仅替换已执行动作的目标；其他动作误差为 0，三个输出的 MSE 为选中动作 MSE 的 1/3。
        target = pred.detach().clone()
        with torch.no_grad():
            # 终止时 y=r；只为非终止样本计算下一观测。消融时在线网络也不追踪梯度。
            td_values = reward.clone()
            nonterminal = ~torch.as_tensor(done, dtype=torch.bool)
            if nonterminal.any():
                bootstrap_model = self.target_model if self.target_model is not None else self.model
                next_values = bootstrap_model(next_state[nonterminal]).amax(dim=1)
                td_values[nonterminal] += self.gamma * next_values
            # 解码 one-hot 动作，仅为实际执行的动作设置 TD 目标。
            target.scatter_(1, action.argmax(dim=1, keepdim=True), td_values.unsqueeze(1))

        # 清除旧梯度，计算平方 TD 误差，反向传播并更新在线网络。
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        # 按优化更新次数计数，单样本更新和一次批量回放均各计一次。
        self.n_updates += 1
        if self.target_model is not None and self.n_updates % self.target_update_interval == 0:
            # 每 C 次优化更新硬同步一次：θ⁻ ← θ。
            self.target_model.load_state_dict(self.model.state_dict())
