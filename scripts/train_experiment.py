"""Train one Snake DQN variant with an exact environment-step budget."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time

from train_upstream import positive_int, save_run_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'snake-ai-pytorch-main'))


def train(args, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from agent import Agent, greedy_action, one_hot_action
    from game import SnakeGameAI

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    exploration_rng = random.Random(f'{args.seed}:exploration')
    replay_rng = random.Random(f'{args.seed}:replay')
    game = SnakeGameAI(render=False, rng=random.Random(f'{args.seed}:environment'))
    agent = Agent(target_update_interval=args.target_update_interval,
                  use_target_network=not args.no_target_network)

    observation = agent.get_state(game)
    fresh_batch = []
    completed = []
    episode_steps = 0
    episode_return = 0
    steps = 0
    started = time.monotonic()
    status = 'completed'
    try:
        for step in range(1, args.steps + 1):
            # 同一步数使用同一探索率，即使各组完成的回合数不同。
            epsilon = 1.0 - 0.95 * min((step - 1) / args.epsilon_decay_steps, 1.0)
            if exploration_rng.random() < epsilon:
                action = one_hot_action(exploration_rng.randrange(3))
            else:
                action = greedy_action(agent.model, observation)
            reward, done, score = game.play_step(action)
            next_observation = agent.get_state(game)
            transition = (observation, action, reward, next_observation, done)
            steps = step
            episode_steps += 1
            episode_return += reward

            if args.no_replay:
                if step > args.warmup_steps:
                    fresh_batch.append(transition)
            else:
                agent.remember(*transition)

            # 两组在相同步数更新一次、使用同样大小的批次。
            # 无回放组按交互顺序使用新样本，更新后丢弃。回放组从经验池均匀抽样。
            after_warmup = step - args.warmup_steps
            if after_warmup > 0 and after_warmup % args.batch_size == 0:
                batch = (fresh_batch if args.no_replay
                         else replay_rng.sample(agent.memory, args.batch_size))
                agent.trainer.train_step(*zip(*batch))
                fresh_batch.clear()

            if done:
                completed.append({
                    'episode': len(completed) + 1, 'env_steps': step,
                    'episode_steps': episode_steps, 'score': score,
                    'episode_return': episode_return,
                    'end_reason': 'collision' if game.is_collision() else 'game_timeout',
                })
                episode_steps = 0
                episode_return = 0
                if step < args.steps:
                    game.reset()
                    next_observation = agent.get_state(game)
            observation = next_observation
            if step % 5000 == 0:
                print(f'{step}/{args.steps} environment steps; '
                      f'{agent.trainer.n_updates} optimizer updates', flush=True)
    except KeyboardInterrupt:
        status = 'interrupted'

    elapsed = time.monotonic() - started
    save_run_checkpoint(agent.model, output)
    with (output / 'scores.csv').open('w', newline='') as file:
        fields = ['episode', 'env_steps', 'episode_steps', 'score', 'episode_return', 'end_reason']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)

    figure, axes = plt.subplots(figsize=(8, 4))
    axes.plot([row['env_steps'] for row in completed], [row['score'] for row in completed])
    axes.set(xlabel='Environment steps', ylabel='Food eaten per episode',
             title='Training scores (completed episodes)', xlim=(0, args.steps))
    figure.tight_layout()
    figure.savefig(output / 'scores.png', dpi=150)
    plt.close(figure)

    variant = 'dqn'
    if args.no_target_network:
        variant += '-no-target'
    if args.no_replay:
        variant += '-no-replay'
    summary = {
        'status': status,
        'variant': variant,
        'seed': args.seed,
        'requested_env_steps': args.steps,
        'env_steps': steps,
        'optimizer_updates': agent.trainer.n_updates,
        'completed_episodes': len(completed),
        'partial_episode_steps': episode_steps,
        'checkpoint_path': str(output / 'model.pth'),
        'config': {
            'target_network': not args.no_target_network,
            'target_update_interval': args.target_update_interval,
            'replay': not args.no_replay,
            'replay_capacity': agent.memory.maxlen if not args.no_replay else 0,
            'warmup_steps': args.warmup_steps,
            'batch_size': args.batch_size,
            'update_every_env_steps': args.batch_size,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': args.epsilon_decay_steps,
            'gamma': agent.gamma,
            'learning_rate': agent.trainer.lr,
            'optimizer': 'Adam',
            'loss': 'MSE over three action outputs',
        },
        'elapsed_seconds': round(elapsed, 3),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return 130 if status == 'interrupted' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=positive_int, required=True, help='total environment steps')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-size', type=positive_int, default=32)
    parser.add_argument('--warmup-steps', type=int, default=256)
    parser.add_argument('--epsilon-decay-steps', type=positive_int, default=10000)
    parser.add_argument('--target-update-interval', type=positive_int, default=100,
                        help='target sync interval measured in optimizer updates')
    parser.add_argument('--no-target-network', action='store_true')
    parser.add_argument('--no-replay', action='store_true')
    args = parser.parse_args()
    from agent import MAX_MEMORY
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be in [0, 2**32)')
    if args.warmup_steps < 0 or args.steps < args.warmup_steps + args.batch_size:
        parser.error('warmup must be nonnegative and the budget must include at least one update')
    if args.batch_size > MAX_MEMORY:
        parser.error(f'--batch-size cannot exceed {MAX_MEMORY}')
    output = ROOT / 'runs' / datetime.now().strftime('experiment-%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    print(f'Experiment outputs: {output}', flush=True)
    return train(args, output)


if __name__ == '__main__':
    raise SystemExit(main())
