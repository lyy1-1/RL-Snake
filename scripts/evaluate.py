"""Compare saved Snake policies and two baselines on the same evaluation seeds."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

from train_upstream import positive_int

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'snake-ai-pytorch-main'))


def rule_action(game):
    """Prefer legal moves that reduce Manhattan distance to the food."""
    from agent import one_hot_action

    actions = [one_hot_action(index) for index in range(3)]
    candidates = []
    for action in actions:
        _, next_head = game.preview_move(action)
        if not game.is_collision(next_head):
            distance = abs(next_head.x - game.food.x) + abs(next_head.y - game.food.y)
            candidates.append((distance, action))
    # Candidate iteration breaks ties straight/right/left. If trapped, go straight.
    return min(candidates, key=lambda candidate: candidate[0])[1] if candidates else actions[0]


def evaluate_episode(game, choose_action, seed, max_steps):
    game.reset(seed=seed)
    policy_rng = random.Random(f'{seed}:evaluation-policy')
    total_reward = 0
    reason = 'evaluation_step_limit'
    for step in range(1, max_steps + 1):
        reward, done, score = game.play_step(choose_action(game, policy_rng))
        total_reward += reward
        if done:
            reason = 'collision' if game.is_collision() else 'game_timeout'
            break
    return {'seed': seed, 'score': score, 'steps': step, 'episode_return': total_reward,
            'end_reason': reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', nargs=2, required=True,
                        metavar=('NAME', 'CHECKPOINT'),
                        help='label and model.pth path; repeat to compare multiple models')
    parser.add_argument('--episodes', type=positive_int, default=30)
    parser.add_argument('--seed-start', type=int, default=10000)
    parser.add_argument('--max-steps', type=positive_int, default=1000, help='limit per evaluated game')
    args = parser.parse_args()
    checkpoints = {}
    for label, path_text in args.model:
        if not label or label in ('random', 'rule') or label in checkpoints:
            parser.error(f'duplicate or reserved policy name: {label!r}')
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            parser.error(f'checkpoint not found: {path}')
        checkpoints[label] = path

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from agent import Agent, greedy_action, one_hot_action
    from game import SnakeGameAI
    from model import Linear_QNet

    torch.set_num_threads(1)
    policies = {
        'random': lambda game, rng: one_hot_action(rng.randrange(3)),
        'rule': lambda game, rng: rule_action(game),
    }
    model_details = {}
    for name, path in checkpoints.items():
        model = Linear_QNet(11, 256, 3)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.eval().requires_grad_(False)
        policies[name] = lambda game, rng, model=model: greedy_action(model, Agent.get_state(game))
        model_details[name] = {'checkpoint': str(path),
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    game = SnakeGameAI(render=False, rng=random.Random(seeds[0]))
    rows = []
    aggregate = {}
    for name, choose_action in policies.items():
        outcomes = [evaluate_episode(game, choose_action, seed, args.max_steps) for seed in seeds]
        rows.extend({'policy': name, **outcome} for outcome in outcomes)
        scores = [outcome['score'] for outcome in outcomes]
        aggregate[name] = {'episodes': len(scores), 'mean_score': statistics.mean(scores),
                           'score_std': statistics.stdev(scores) if len(scores) > 1 else None,
                           'max_score': max(scores),
                           'mean_steps': statistics.mean(outcome['steps'] for outcome in outcomes)}

    output = ROOT / 'runs' / datetime.now().strftime('evaluation-%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    with (output / 'episodes.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'episode_seeds': seeds, 'max_steps_per_episode': args.max_steps,
               'model_weights': model_details, 'aggregate': aggregate}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    figure, axes = plt.subplots(figsize=(8, 4.5))
    deviations = None if len(seeds) == 1 else [entry['score_std'] for entry in aggregate.values()]
    axes.bar(list(aggregate), [entry['mean_score'] for entry in aggregate.values()],
             yerr=deviations, capsize=4)
    axes.set(ylabel='Food eaten per episode', title=f'Policy evaluation on {len(seeds)} shared seeds')
    axes.tick_params(axis='x', labelrotation=15)
    figure.tight_layout()
    figure.savefig(output / 'evaluation.png', dpi=160)
    plt.close(figure)

    print(f'Evaluation outputs: {output}', flush=True)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
