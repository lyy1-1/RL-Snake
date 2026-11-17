from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from video_recorder import TailVideoRecorder


class VideoRecorderTests(unittest.TestCase):
    def test_mp4_contains_only_last_150_steps_at_10_fps(self):
        try:
            import imageio_ffmpeg
        except ImportError:
            self.skipTest('install requirements-video.txt to test MP4 encoding')
        recorder = TailVideoRecorder(max_frames=150, fps=10)
        for step in range(155):
            recorder.append(np.full((16, 16, 3), step, dtype=np.uint8))
        self.assertEqual(len(recorder.frames), 150)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tail.mp4'
            result = recorder.save(path)
            self.assertEqual(result['first_step'], 6)
            self.assertEqual(result['last_step'], 155)
            self.assertEqual(result['duration_seconds'], 15)
            reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
            try:
                metadata = next(reader)
                decoded = [np.frombuffer(frame, dtype=np.uint8).mean() for frame in reader]
            finally:
                reader.close()
            self.assertEqual(metadata['fps'], 10)
            self.assertAlmostEqual(metadata['duration'], 15, places=1)
            self.assertEqual(len(decoded), 150)
            self.assertAlmostEqual(decoded[0], 5, delta=3)
            self.assertAlmostEqual(decoded[-1], 154, delta=3)
            self.assertFalse(path.with_name('tail.part.mp4').exists())

    def test_short_game_keeps_all_available_frames(self):
        recorder = TailVideoRecorder(max_frames=150, fps=10)
        for step in range(4):
            recorder.append(np.full((16, 16, 3), step, dtype=np.uint8))
        self.assertEqual(len(recorder.frames), 4)
        self.assertEqual(recorder.steps, 4)

    def test_frame_is_copied_before_reused_render_buffer_changes(self):
        recorder = TailVideoRecorder()
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        recorder.append(frame)
        frame.fill(255)
        self.assertEqual(recorder.frames[0].max(), 0)

    def test_empty_buffer_does_not_create_a_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.mp4'
            self.assertIsNone(TailVideoRecorder().save(path))
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
