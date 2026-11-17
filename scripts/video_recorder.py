"""Keep the last N rendered game steps and encode them as an MP4."""

from collections import deque
from pathlib import Path

import numpy as np


class TailVideoRecorder:
    def __init__(self, max_frames=150, fps=10):
        if max_frames <= 0 or fps <= 0:
            raise ValueError('max_frames and fps must be positive')
        self.frames = deque(maxlen=max_frames)
        self.fps = fps
        self.steps = 0

    def append(self, frame):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('expected an RGB uint8 image with shape (height, width, 3)')
        self.frames.append(np.ascontiguousarray(frame).copy())
        self.steps += 1

    def save(self, path):
        if not self.frames:
            return None
        import imageio_ffmpeg

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + '.part.mp4')
        height, width = self.frames[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(
            str(temporary), size=(width, height), fps=self.fps,
            codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
            macro_block_size=2, ffmpeg_log_level='error',
            output_params=['-movflags', '+faststart'])
        try:
            writer.send(None)
            for frame in self.frames:
                writer.send(frame)
        finally:
            writer.close()
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError('MP4 encoder produced no video')
        temporary.replace(path)
        return {'path': str(path), 'frame_count': len(self.frames),
                'fps': self.fps, 'duration_seconds': len(self.frames) / self.fps,
                'first_step': self.steps - len(self.frames) + 1,
                'last_step': self.steps}
