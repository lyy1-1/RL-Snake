"""Play a saved Snake Q network without training or random exploration."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=positive_int, default=3)
    parser.add_argument('--max-steps', type=positive_int, default=300)
    parser.add_argument('--fps', type=int, default=10, help='steps per second; 0 means unlimited')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--headless', action='store_true', help='run without a game window')
    parser.add_argument('--wait', action='store_true', help='wait for Space before playing, for screen recording')
    parser.add_argument('--tail-steps', type=positive_int, default=150,
                        help='retain the last N steps of each episode in its MP4 (default: 150)')
    parser.add_argument('--video-fps', type=positive_int, default=10,
                        help='MP4 playback frame rate, independent of game speed (default: 10)')
    parser.add_argument('--no-video', action='store_true', help='disable automatic MP4 recording')
    args = parser.parse_args()
    if args.fps < 0:
        parser.error('--fps must be nonnegative')
    if args.headless and args.wait:
        parser.error('--wait requires a visible window')
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f'checkpoint not found: {checkpoint}')
    if not args.no_video:
        try:
            import imageio_ffmpeg
            imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            parser.error(f'MP4 encoding unavailable: {exc}. Install requirements-video.txt or use --no-video')
    root = Path(__file__).resolve().parents[1]
    output = root / 'runs' / ('demo-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True)
    video_output = root / 'runs' / 'videos' / output.name
    if args.headless:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(root / 'snake-ai-pytorch-main'))

    import numpy as np
    import pygame
    import torch
    import game as game_module
    from agent import Agent, greedy_action
    from video_recorder import TailVideoRecorder

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent = Agent()
    agent.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    agent.model.eval()
    agent.model.requires_grad_(False)
    game_module.SPEED = args.fps
    game = game_module.SnakeGameAI()
    game.display = pygame.display.set_mode((game.w, game.h + 28))
    pygame.display.set_caption('RL-Snake: trained model demo')
    font = pygame.font.Font(None, 24)
    results = []
    status = 'completed'
    recorder = None
    episode = 0
    step = 0
    score = 0

    def finish_episode(reason):
        nonlocal recorder
        video = None
        if recorder is not None:
            video = recorder.save(video_output / f'episode-{episode:03d}.mp4')
        result = {'episode': episode, 'score': score, 'steps': step,
                  'end_reason': reason, 'video': video}
        results.append(result)
        recorder = None
        print(json.dumps(result), flush=True)

    print(f'Demo only, no training; model: {checkpoint}; records: {output}', flush=True)
    if not args.no_video:
        print(f'MP4: last {args.tail_steps} steps per game at {args.video_fps} fps; {video_output}', flush=True)
    try:
        if args.wait:
            game._update_ui()
            game.display.blit(font.render('Start recording, then press SPACE', True, (255, 255, 255)), (5, game.h + 2))
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        waiting = False
                game.clock.tick(30)
        for episode in range(1, args.episodes + 1):
            step, score = 0, 0
            recorder = None if args.no_video else TailVideoRecorder(args.tail_steps, args.video_fps)
            if episode > 1:
                game.reset()
            for step in range(1, args.max_steps + 1):
                # 评估使用贪心策略，不调用带随机探索的 get_action，也不更新参数。
                action = greedy_action(agent.model, agent.get_state(game))
                _, done, score = game.play_step(action)
                game._update_ui()
                label = f'Saved model | Greedy | Game {episode} | Step {step} | Score {score}'
                pygame.draw.rect(game.display, (0, 0, 0), (0, game.h, game.w, 28))
                game.display.blit(font.render(label, True, (255, 255, 255)), (5, game.h + 2))
                pygame.display.flip()
                if recorder is not None:
                    frame = pygame.surfarray.array3d(game.display).transpose(1, 0, 2)
                    recorder.append(frame)
                if done:
                    reason = 'collision' if game.is_collision() else 'game_timeout'
                    break
            else:
                reason = 'demo_step_limit'
            finish_episode(reason)
    except KeyboardInterrupt:
        status = 'interrupted'
    except SystemExit:
        status = 'window_closed'
    except Exception:
        status = 'error'
        raise
    finally:
        try:
            if status in ('interrupted', 'window_closed') and recorder is not None and recorder.frames:
                finish_episode(status)
        finally:
            pygame.quit()
        summary = {'status': status, 'checkpoint': str(checkpoint),
                   'checkpoint_sha256': checkpoint_hash, 'seed': args.seed,
                   'policy': 'greedy', 'training_updates': agent.trainer.n_updates,
                   'max_steps_per_episode': args.max_steps, 'fps': args.fps,
                   'video_enabled': not args.no_video, 'tail_steps': args.tail_steps,
                   'video_fps': args.video_fps,
                   'results': results}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'Demo finished; summary: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
