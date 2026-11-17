# RL-Snake

研究生强化学习课程项目，已完成上游原版运行验证；按用户授权，当前先实现**目标网络与固定 TD 目标**，其余改造和实验参数仍待审核。

## 项目方向

以 Mnih 等人的 *Human-level control through deep reinforcement learning*（Nature, 2015）为主要理论依据，研究 DQN 关键机制在贪吃蛇任务中的迁移与验证。课程项目将贯通 **MDP 建模 → 算法学习 → 实验效果**。

贪吃蛇环境及其具体实现属于本项目的工程设计，不是原论文的实验。项目已选择优先复用 patrickloeber 的开源贪吃蛇代码，并做满足课程研究目标所需的最小改造。先前自建有序网格观察与轻量 CNN 的方案已被这一选择替代；修订后的完整技术方案仍待审核。按用户要求，已先用独立环境跑通未修改的原版，完成 3 个回合及训练和回放更新。

## 文档阅读顺序

1. [00. 小组协作与提交规范](docs/00.小组协作与提交规范.md)：加入仓库、分支、提交与 PR 流程。
2. [项目方案决策记录](docs/PROJECT_DECISIONS.md)：已确认、待确认和暂定的方案状态。此文件保留先前指定的固定路径。
3. [01. 技术方案（待审核）](docs/01.技术方案-待审核.md)：具体环境、MDP、DQN、实验与交付提案，审核通过前不实施。
4. [02. 开源代码复用评估](docs/02.开源代码复用评估.md)：已选择以第一份开源实现为主要代码基础；详见复用边界与必需修改。
5. [03. 课程 PPT 符合性复核](docs/03.课程PPT符合性复核.md)：课件与上游源码的逐项静态对照。
6. [04. 原版运行与环境配置](docs/04.原版运行与环境配置.md)：依赖版本、启动命令与实际验证结果。
7. [05. 强化学习知识与代码对应](docs/05.强化学习知识与代码对应.md)：从 MDP、Q-learning 到 DQN 和实验验证的学习说明。
8. [06. 目标网络与固定 TD 目标](docs/06.目标网络与固定TD目标.md)：第一步算法改动、暂定同步间隔和验证方式。
9. [07. 中期项目汇报内容确认](docs/07.中期项目汇报内容确认.md)：1–2 分钟汇报草稿、4 人分工与待确认事项。

后续新增说明文档将按数字前缀继续排列。

## 资料

- 主论文：[Human-level control through deep reinforcement learning](Human-level%20control%20through%20deep%20reinforcement%20learning.pdf)
- 参考代码：[patrickloeber/snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch)
- 参考代码：[billodalroy/Snake-Game-Deep-Reinforcement-Learning](https://github.com/billodalroy/Snake-Game-Deep-Reinforcement-Learning)

参考代码用于了解实现思路；算法与论文的对应关系以论文正文和实际代码为准。

## 协作状态

本地已建立 `.venv` 并完成原版验证。2026-11-16 按用户授权实现目标网络与固定 TD 目标；2026-11-17 将 `snake-ai-pytorch-main/` 中的改造代码、启动脚本、测试和文档一并纳入 Git，保留上游 MIT 许可。字体与原版 ZIP 留在本地，克隆仓库后会自动使用 Pygame 内置字体。环境说明见 [04. 运行与环境配置](docs/04.原版运行与环境配置.md)，改动见 [06. 实现说明](docs/06.目标网络与固定TD目标.md)。其余改造与正式实验仍待审核；以后仍按用户明确要求再推送。

目前推荐在项目根目录通过独立启动脚本训练，以避免本机原生绘图窗口崩溃，并指定回合数：

```bash
.venv/bin/python scripts/train_upstream.py --episodes 100
```

游戏窗口保留，曲线与每回合得分保存在启动时打印的 `runs/upstream-时间戳/` 目录。详见运行文档。
