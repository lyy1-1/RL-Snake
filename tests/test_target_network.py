"""Numerical checks for the project's first DQN mechanism change."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'snake-ai-pytorch-main'))
from model import Linear_QNet, QTrainer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_upstream import save_run_checkpoint


class CaptureMSE(nn.Module):
    def forward(self, target, prediction):
        self.target = target
        self.prediction = prediction
        return nn.functional.mse_loss(prediction, target)


class TargetNetworkTests(unittest.TestCase):
    def make_trainer(self, interval=1000):
        network = nn.Linear(2, 3)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.copy_(torch.tensor([8., 1., 0.]))
        return QTrainer(network, lr=0.01, gamma=0.9, target_update_interval=interval)

    def test_initial_copy_is_independent_frozen_and_not_optimized(self):
        trainer = QTrainer(Linear_QNet(11, 256, 3), lr=0.001, gamma=0.9)
        self.assertFalse(trainer.target_model.training)
        for online, target in zip(trainer.model.parameters(), trainer.target_model.parameters()):
            self.assertTrue(torch.equal(online, target))
            self.assertNotEqual(online.data_ptr(), target.data_ptr())
            self.assertFalse(target.requires_grad)
        optimized = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        self.assertEqual(optimized, {id(p) for p in trainer.model.parameters()})
        self.assertTrue(optimized.isdisjoint({id(p) for p in trainer.target_model.parameters()}))

    def test_batch_targets_use_target_network_and_terminal_reward_only(self):
        trainer = self.make_trainer()
        with torch.no_grad():
            trainer.target_model.bias.copy_(torch.tensor([2., 4., 1.]))
        capture = CaptureMSE()
        trainer.criterion = capture
        output_grad_flags = []
        hook = trainer.target_model.register_forward_hook(
            lambda module, inputs, output: output_grad_flags.append(output.requires_grad))
        trainer.train_step([[0., 0.], [0., 0.]], [[1, 0, 0], [0, 0, 1]],
                           [1., -10.], [[0., 0.], [0., 0.]], [False, True])
        hook.remove()
        # First target is 1 + .9 * 4 = 4.6, rather than 1 + .9 * 8.
        torch.testing.assert_close(capture.target, torch.tensor([[4.6, 1., 0.], [8., 1., -10.]]))
        self.assertEqual(output_grad_flags, [False])
        self.assertFalse(capture.target.requires_grad)
        self.assertIsNone(capture.target.grad_fn)
        self.assertTrue(capture.prediction.requires_grad)
        self.assertTrue(all(p.grad is None for p in trainer.target_model.parameters()))
        # Stop-gradient MSE derivative: 2 * (prediction - target) / six entries.
        torch.testing.assert_close(trainer.model.bias.grad, torch.tensor([6.8 / 6, 0., 20. / 6]))
        self.assertLess(trainer.model.bias[0].item(), 8.)
        self.assertLess(trainer.model.bias[2].item(), 0.)

    def test_single_terminal_sample_never_evaluates_next_state(self):
        trainer = self.make_trainer()
        capture = CaptureMSE()
        trainer.criterion = capture
        def forbidden(*args):
            raise AssertionError('Terminal transitions must not bootstrap')
        hook = trainer.target_model.register_forward_pre_hook(forbidden)
        trainer.train_step([0., 0.], [0, 1, 0], -10., [float('nan'), float('nan')], True)
        hook.remove()
        torch.testing.assert_close(capture.target, torch.tensor([[8., -10., 0.]]))
        self.assertEqual(trainer.n_updates, 1)

    def test_sync_after_exact_number_of_optimizer_updates_including_replay(self):
        trainer = self.make_trainer(interval=2)
        before = {key: value.clone() for key, value in trainer.target_model.state_dict().items()}
        trainer.train_step([0., 0.], [1, 0, 0], -10., [0., 0.], True)
        self.assertEqual(trainer.n_updates, 1)
        for key, value in trainer.target_model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertFalse(torch.equal(trainer.model.bias, trainer.target_model.bias))
        # A batch update counts once, independently of its number of samples.
        trainer.train_step([[0., 0.], [0., 0.]], [[1, 0, 0], [0, 1, 0]],
                           [-10., -10.], [[0., 0.], [0., 0.]], [True, True])
        self.assertEqual(trainer.n_updates, 2)
        for key, value in trainer.target_model.state_dict().items():
            self.assertTrue(torch.equal(value, trainer.model.state_dict()[key]))
        self.assertFalse(trainer.target_model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in trainer.target_model.parameters()))

    def test_invalid_sync_intervals(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_trainer(interval=value)

    def test_run_checkpoint_is_isolated_from_shared_weights(self):
        trainer = self.make_trainer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / 'model.pth'
            shared.write_bytes(b'existing trained weights')
            run_dir = root / 'run'
            run_dir.mkdir()
            save_run_checkpoint(trainer.model, run_dir)
            self.assertEqual(shared.read_bytes(), b'existing trained weights')
            saved = torch.load(run_dir / 'model.pth', weights_only=True)
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(saved[key], value)


if __name__ == '__main__':
    unittest.main()
