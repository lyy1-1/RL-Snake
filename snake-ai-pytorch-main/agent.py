import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
TARGET_UPDATE_INTERVAL = 1000  # Optimizer steps; provisional project default.

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        # γ 是折扣因子，决定未来奖励在回报中的权重。
        self.gamma = 0.9 # discount rate
        # 经验回放池：保存历史交互，容量满后丢弃最早经验。
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # 用神经网络近似 Q 值：11 项观察输入，输出直行、右转、左转的价值。
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma,
                                target_update_interval=TARGET_UPDATE_INTERVAL)


    def get_state(self, game):
        # 这里返回局部观察 o：危险 3 项、朝向 4 项、食物方位 4 项。
        # 观察缺少完整蛇身和步数，不能保证满足马尔可夫性。
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # 前方是否危险
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # 右方是否危险
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # 左方是否危险
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # 当前朝向
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # 食物相对方位
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # 存储一步经验：(观察、动作、奖励、下一观察、结束标记)。
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        # 均匀随机回放历史经验，减弱时间相关性并复用数据；仍使用一步 TD 目标。
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # 每步用当前新经验做一次 TD 更新；回合结束时另外做一次回放更新。
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # ε-greedy：随机探索，或利用当前估值选择最大 Q 值动作。
        # 此处 epsilon 是整数阈值：初始随机分支概率为 80/201，80 局后降为 0。
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # 贪心动作由在线网络选择，目标网络只参与训练目标计算。
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    # 交互学习流程：观察 → 选动作 → 获得奖励与下一观察 → TD 更新 → 保存经验。
    # 行为策略包含探索，学习目标取下一状态最大 Q 值，因此 Q-learning 是 off-policy。
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        # reward 是学习反馈，score 是吃到的食物数。
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 一次 episode 结束后重置环境，网络参数和历史经验继续保留。
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            # 训练累计平均分用于观察学习过程，实验结论还需要独立评估和多种子统计。
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
