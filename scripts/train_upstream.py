"""Run the upstream-based trainer with a finite episode budget and file plots."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time


class TrainingComplete(Exception):
    pass


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number


def save_run_checkpoint(model, output_dir, file_name='model.pth'):
    import torch
    torch.save(model.state_dict(), output_dir / file_name)


def main():
    # episodes 是游戏回合数，target-update-interval 是目标网络同步的优化更新间隔。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=positive_int, default=100,
                        help='number of completed games (default: 100)')
    parser.add_argument('--max-seconds', type=positive_int,
                        help='optional wall time budget for the training loop')
    parser.add_argument('--target-update-interval', type=positive_int, default=1000,
                        help='hard-sync target after this many optimizer steps (default: 1000)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'snake-ai-pytorch-main'
    if not (source / 'agent.py').is_file():
        parser.error('missing snake-ai-pytorch-main/agent.py; see docs/04.原版运行与环境配置.md')
    output = root / 'runs' / ('upstream-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True)

    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    import pygame

    os.chdir(source)  # Upstream loads arial.ttf using a relative path.
    sys.path.insert(0, str(source))
    import agent
    agent.TARGET_UPDATE_INTERVAL = args.target_update_interval

    def save_record(model, file_name='model.pth'):
        save_run_checkpoint(model, output, file_name)
    agent.Linear_QNet.save = save_record

    completed = 0
    started = time.monotonic()
    status = 'error'
    figure, axes = plt.subplots()

    def record(scores, means):
        # 记录训练得分与累计均值；独立评估还需固定评估策略并停止参数更新。
        nonlocal completed
        completed = len(scores)
        with (output / 'scores.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['episode', 'score', 'mean_score'])
            writer.writerows(zip(range(1, completed + 1), scores, means))
        axes.clear()
        axes.plot(scores, label='Score')
        axes.plot(means, label='Mean score')
        axes.set(xlabel='Completed games', ylabel='Food eaten', title='Upstream training')
        axes.legend()
        figure.savefig(output / 'scores.png')
        if completed >= args.episodes:
            raise TrainingComplete()

    def time_limit(signum, frame):
        # 达到运行时限只停止采样，不生成终止经验或额外奖励。
        raise TimeoutError('Training wall time budget reached')

    agent.plot = record
    old_handler = None
    if args.max_seconds:
        if not hasattr(signal, 'SIGALRM'):
            parser.error('--max-seconds currently requires macOS or Linux')
        old_handler = signal.signal(signal.SIGALRM, time_limit)
        signal.alarm(args.max_seconds)
    print(f'Target: {args.episodes} games; target-network sync: '
          f'{args.target_update_interval} optimizer steps; output: {output}', flush=True)
    try:
        agent.train()
    except TrainingComplete:
        status = 'completed'
    except TimeoutError:
        status = 'time_limit'
    except KeyboardInterrupt:
        status = 'interrupted'
    except SystemExit:
        status = 'window_closed'
    finally:
        if args.max_seconds:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        pygame.quit()
        plt.close('all')
        summary = {'status': status, 'requested_episodes': args.episodes,
                   'variant': 'upstream-with-target-network',
                   'target_update_interval': args.target_update_interval,
                   'checkpoint_path': str(output / 'model.pth') if (output / 'model.pth').exists() else None,
                   'completed_episodes': completed,
                   'elapsed_seconds': round(time.monotonic() - started, 2),
                   'plot_backend': matplotlib.get_backend()}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    return 0 if status != 'error' else 1


if __name__ == '__main__':
    raise SystemExit(main())
