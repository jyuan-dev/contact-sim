"""
Tests for src/training/trainer.py.
Covers: TeeLogger (write/flush/close) and BaseTrainer (init, log_scalar, log_image,
        save_checkpoint, close) using a real temporary directory.
"""

import io
import os
import sys
import tempfile
import unittest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.training.trainer import TeeLogger, BaseTrainer


class TestTeeLogger(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self._tmpdir.name, "test.log")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_write_to_file_and_terminal(self):
        """TeeLogger should write to both the file and the captured terminal stream."""
        fake_terminal = io.StringIO()
        logger = TeeLogger.__new__(TeeLogger)
        logger.terminal = fake_terminal
        logger.log_file = open(self.log_path, 'w')

        logger.write("hello world")
        logger.log_file.close()

        self.assertIn("hello world", fake_terminal.getvalue())
        with open(self.log_path) as f:
            self.assertIn("hello world", f.read())

    def test_flush_does_not_raise(self):
        """flush() should execute without raising."""
        logger = TeeLogger(self.log_path)
        # Temporarily restore stdout so TeeLogger doesn't interfere
        original_stdout = sys.stdout
        try:
            logger.flush()
        finally:
            sys.stdout = original_stdout
            logger.close()

    def test_close_idempotent(self):
        """Calling close() twice should not raise."""
        logger = TeeLogger(self.log_path)
        original_stdout = sys.stdout
        try:
            logger.close()
            logger.close()   # second call should be a no-op
        finally:
            sys.stdout = original_stdout


class TestBaseTrainer(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.save_dir = self._tmpdir.name
        self._original_stdout = sys.stdout

    def tearDown(self):
        # Restore stdout in case any test left it redirected
        sys.stdout = self._original_stdout
        self._tmpdir.cleanup()

    def _make_trainer(self):
        return BaseTrainer(
            save_dir=self.save_dir,
            experiment_name="unit_test_exp"
        )

    def test_directories_created(self):
        """Initialising BaseTrainer should create save_dir and tb_logs/."""
        trainer = self._make_trainer()
        trainer.close()
        self.assertTrue(os.path.isdir(self.save_dir))
        self.assertTrue(os.path.isdir(os.path.join(self.save_dir, 'tb_logs')))

    def test_log_file_created(self):
        """A train.log file should be created in save_dir."""
        trainer = self._make_trainer()
        trainer.close()
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir, 'train.log')))

    def test_log_scalar_does_not_raise(self):
        """log_scalar should call TensorBoard without raising."""
        trainer = self._make_trainer()
        try:
            trainer.log_scalar("loss/train", 0.42, global_step=1)
        finally:
            trainer.close()

    def test_log_image_does_not_raise(self):
        """log_image should call TensorBoard with a [C, H, W] image tensor."""
        trainer = self._make_trainer()
        try:
            img = torch.rand(3, 32, 32)
            trainer.log_image("vis/frame", img, global_step=1)
        finally:
            trainer.close()

    def test_save_checkpoint(self):
        """save_checkpoint should persist a .pt file and return its path."""
        trainer = self._make_trainer()
        try:
            state = {'epoch': 1, 'loss': 0.5}
            ckpt_path = trainer.save_checkpoint(state, filename="test_ckpt.pt")
        finally:
            trainer.close()

        self.assertTrue(os.path.isfile(ckpt_path))
        loaded = torch.load(ckpt_path, weights_only=True)
        self.assertEqual(loaded['epoch'], 1)
        self.assertAlmostEqual(loaded['loss'], 0.5)

    def test_close_restores_stdout(self):
        """After close(), sys.stdout should be restored to the original stream."""
        trainer = self._make_trainer()
        trainer.close()
        # sys.stdout should no longer be the TeeLogger
        self.assertNotIsInstance(sys.stdout, TeeLogger)

    def test_multiple_trainers_sequential(self):
        """Two sequentially created BaseTrainer instances should not conflict."""
        save_dir_1 = os.path.join(self.save_dir, "exp1")
        save_dir_2 = os.path.join(self.save_dir, "exp2")

        trainer1 = BaseTrainer(save_dir=save_dir_1, experiment_name="exp1")
        trainer1.close()

        trainer2 = BaseTrainer(save_dir=save_dir_2, experiment_name="exp2")
        trainer2.close()

        self.assertTrue(os.path.isfile(os.path.join(save_dir_1, 'train.log')))
        self.assertTrue(os.path.isfile(os.path.join(save_dir_2, 'train.log')))


if __name__ == '__main__':
    unittest.main()
