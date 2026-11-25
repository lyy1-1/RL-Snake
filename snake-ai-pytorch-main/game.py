import pygame
import random
from enum import Enum
from collections import namedtuple
from pathlib import Path
import numpy as np

# 游戏环境提供 MDP 的转移 P、奖励 R、初始状态和结束规则。

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480, *, render=True, rng=None):
        self.w = w
        self.h = h
        self.render = render
        self.rng = rng if rng is not None else random
        if render:
            pygame.init()
            font_path = Path(__file__).with_name('arial.ttf')
            self.font = pygame.font.Font(str(font_path) if font_path.is_file() else None, 25)
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        # 完整状态包括有序蛇身、朝向、食物和步数；棋盘尺寸为固定参数。
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        # 食物位置引入转移随机性：随机抽格，若被蛇占据则重新抽样。
        x = self.rng.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = self.rng.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        # 执行动作 a_t，返回奖励 r_(t+1)、结束标记和得分；下一观察由 Agent 提取。
        self.frame_iteration += 1
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # 普通移动奖励为 0，碰撞或超时奖励为 -10，吃到食物奖励为 +10。
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            # 当前碰撞和超时共用结束标记，尚未区分任务终止与时间截断。
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        self.direction, self.head = self.preview_move(action)

    def preview_move(self, action):
        """Return the new direction and head position without changing the game."""
        # 动作空间 A：one-hot 编码 [1,0,0]直行、[0,1,0]右转、[0,0,1]左转。

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        x = self.head.x
        y = self.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE

        return new_dir, Point(x, y)
