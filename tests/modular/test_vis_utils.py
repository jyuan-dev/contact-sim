"""
Tests for src/utils/vis_utils.py.
Covers: render_slot_overlay_frame, save_frames_to_gif, SLOT_COLORS_RGB, GT_COLORS_RGB.
"""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.utils.vis_utils import (
    SLOT_COLORS_RGB,
    GT_COLORS_RGB,
    render_slot_overlay_frame,
    save_frames_to_gif,
)


class TestSlotColorsRGB(unittest.TestCase):
    """Validate the slot color palette constants."""

    def test_all_colors_are_valid_rgb_tuples(self):
        """Every slot color must be a 3-tuple of ints in [0, 255]."""
        for slot_idx, color in SLOT_COLORS_RGB.items():
            self.assertEqual(len(color), 3)
            for c in color:
                self.assertIsInstance(c, int)
                self.assertGreaterEqual(c, 0)
                self.assertLessEqual(c, 255)

    def test_has_colors_for_expected_slots(self):
        """Default palette must cover at least 3 slots."""
        self.assertGreaterEqual(len(SLOT_COLORS_RGB), 3)
        for k in (0, 1, 2):
            self.assertIn(k, SLOT_COLORS_RGB)


class TestGTColorsRGB(unittest.TestCase):
    """Validate the ground-truth color palette constants."""

    def test_all_colors_are_valid_rgb_tuples(self):
        """Every GT color must be a 3-tuple of ints in [0, 255]."""
        for class_idx, color in GT_COLORS_RGB.items():
            self.assertEqual(len(color), 3)
            for c in color:
                self.assertIsInstance(c, int)
                self.assertGreaterEqual(c, 0)
                self.assertLessEqual(c, 255)

    def test_has_colors_for_expected_classes(self):
        """Default palette must cover at least 3 classes."""
        self.assertGreaterEqual(len(GT_COLORS_RGB), 3)
        for k in (0, 1, 2):
            self.assertIn(k, GT_COLORS_RGB)


class TestRenderSlotOverlayFrame(unittest.TestCase):
    """Tests for the composite slot-overlay frame renderer."""

    def _make_frame_rgb(self, H=64, W=64):
        """Create a medium-gray RGB frame in uint8."""
        return np.full((H, W, 3), 128, dtype=np.uint8)

    def _make_pred_masks(self, K=3, H=64, W=64):
        """Create hard one-hot prediction masks [K, H, W]."""
        masks = np.zeros((K, H, W), dtype=np.float32)
        strip = H // K
        for k in range(K):
            masks[k, k * strip:(k + 1) * strip, :] = 1.0
        return masks

    def _make_gt_masks(self, M=3, H=64, W=64):
        """Create hard ground-truth masks [M, H, W]."""
        masks = np.zeros((M, H, W), dtype=np.float32)
        strip = H // M
        for m in range(M):
            masks[m, m * strip:(m + 1) * strip, :] = 1.0
        return masks

    # ── Basic return-type / shape tests ───────────────────────────────────────

    def test_output_is_uint8(self):
        """Output must be a uint8 numpy array."""
        frame = self._make_frame_rgb()
        masks = self._make_pred_masks()
        out = render_slot_overlay_frame(frame, masks)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.uint8)

    def test_output_shape(self):
        """Output must be resized to (240, 480, 3) — H=240, W=480."""
        out = render_slot_overlay_frame(self._make_frame_rgb(), self._make_pred_masks())
        self.assertEqual(out.shape, (240, 480, 3))

    def test_banner_text_included(self):
        """When banner_text is provided, the top pixels should differ from background."""
        frame = self._make_frame_rgb()
        masks = self._make_pred_masks()
        no_banner = render_slot_overlay_frame(frame, masks, banner_text="")
        with_banner = render_slot_overlay_frame(frame, masks, banner_text="Test")
        # Top row should differ when banner is present
        self.assertFalse(np.allclose(no_banner[0, :, :], with_banner[0, :, :]))

    # ── GT mask rendering ─────────────────────────────────────────────────────

    def test_render_with_gt_masks(self):
        """Rendering with GT masks should not raise."""
        out = render_slot_overlay_frame(
            self._make_frame_rgb(),
            self._make_pred_masks(),
            gt_masks_t=self._make_gt_masks(),
        )
        self.assertEqual(out.shape, (240, 480, 3))

    def test_render_with_none_gt_masks(self):
        """Render with gt_masks_t=None should not raise."""
        out = render_slot_overlay_frame(
            self._make_frame_rgb(), self._make_pred_masks(), gt_masks_t=None
        )
        self.assertEqual(out.shape, (240, 480, 3))

    def test_render_with_empty_gt_masks(self):
        """Render with all-zero GT masks should not raise."""
        out = render_slot_overlay_frame(
            self._make_frame_rgb(),
            self._make_pred_masks(),
            gt_masks_t=np.zeros((3, 64, 64), dtype=np.float32),
        )
        self.assertEqual(out.shape, (240, 480, 3))

    # ── Input size handling ───────────────────────────────────────────────────

    def test_non_64x64_input_resized(self):
        """Non-64x64 input frames should be resized to 64x64 by the renderer.
        Masks must match the frame's spatial dimensions."""
        frame = np.full((128, 128, 3), 128, dtype=np.uint8)
        # The function resizes the frame to 64x64 but masks must already be at 64x64
        # (matching how the model outputs masks at its internal resolution)
        masks = self._make_pred_masks(K=3, H=64, W=64)
        out = render_slot_overlay_frame(frame, masks)
        self.assertEqual(out.shape, (240, 480, 3))

    def test_single_slot(self):
        """Single-slot mask should render without error."""
        masks = np.ones((1, 64, 64), dtype=np.float32)
        out = render_slot_overlay_frame(self._make_frame_rgb(), masks)
        self.assertEqual(out.shape, (240, 480, 3))

    def test_many_slots(self):
        """Many-slot mask should render without error."""
        masks = np.random.rand(10, 64, 64).astype(np.float32)
        out = render_slot_overlay_frame(self._make_frame_rgb(), masks)
        self.assertEqual(out.shape, (240, 480, 3))

    # ── Color correctness ─────────────────────────────────────────────────────

    def test_output_is_valid_rgb(self):
        """All pixel values in output must be in [0, 255]."""
        out = render_slot_overlay_frame(
            self._make_frame_rgb(), self._make_pred_masks(), self._make_gt_masks()
        )
        self.assertGreaterEqual(out.min(), 0)
        self.assertLessEqual(out.max(), 255)

    def test_left_right_split(self):
        """Left half should not equal right half (GT vs slot overlay)."""
        out = render_slot_overlay_frame(
            self._make_frame_rgb(), self._make_pred_masks(), self._make_gt_masks()
        )
        left = out[:, :240, :]
        right = out[:, 240:, :]
        self.assertFalse(np.allclose(left, right),
                         "Left (GT) and right (slots) panels should differ")


class TestSaveFramesToGIF(unittest.TestCase):
    """Tests for the animated GIF writer."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.out_path = os.path.join(self._tmpdir.name, "test.gif")

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_frames(self, n=3, H=64, W=64):
        """Create n distinct RGB frames."""
        frames = []
        for i in range(n):
            f = np.full((H, W, 3), 50 + i * 50, dtype=np.uint8)
            frames.append(f)
        return frames

    def test_saves_gif_file(self):
        """save_frames_to_gif should create a .gif file."""
        frames = self._make_frames(3)
        save_frames_to_gif(frames, self.out_path, fps=10)
        self.assertTrue(os.path.isfile(self.out_path))

    def test_gif_is_valid(self):
        """Saved GIF should be openable by PIL and contain correct number of frames."""
        frames = self._make_frames(5)
        save_frames_to_gif(frames, self.out_path, fps=10)
        img = Image.open(self.out_path)
        self.assertEqual(img.format, "GIF")
        n_frames = 0
        try:
            while True:
                n_frames += 1
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        self.assertEqual(n_frames, 5)

    def test_single_frame(self):
        """Single-frame GIF should be created without error."""
        frames = self._make_frames(1)
        save_frames_to_gif(frames, self.out_path, fps=10)
        self.assertTrue(os.path.isfile(self.out_path))

    def test_empty_list_no_error(self):
        """Empty frame list should not raise or create an invalid file
        (the function simply returns after the list check)."""
        save_frames_to_gif([], self.out_path)
        # No file should be created for empty list
        self.assertFalse(os.path.isfile(self.out_path))

    def test_creates_parent_directory(self):
        """save_frames_to_gif should create missing parent directories."""
        nested_path = os.path.join(self._tmpdir.name, "subdir", "nested.gif")
        frames = self._make_frames(2)
        save_frames_to_gif(frames, nested_path, fps=7)
        self.assertTrue(os.path.isfile(nested_path))

    def test_custom_fps(self):
        """Different FPS values should produce valid GIFs."""
        for fps in (1, 5, 15, 30):
            path = os.path.join(self._tmpdir.name, f"test_{fps}.gif")
            save_frames_to_gif(self._make_frames(2), path, fps=fps)
            self.assertTrue(os.path.isfile(path))

    def test_roundtrip_colors_are_rgb(self):
        """Frames written to GIF and read back should have RGB-order channels,
        not BGR-swapped."""
        # Create distinctive R, G, B frames
        frames = [
            np.full((32, 32, 3), [255, 0, 0], dtype=np.uint8),   # pure red
            np.full((32, 32, 3), [0, 255, 0], dtype=np.uint8),    # pure green
            np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8),    # pure blue
        ]
        save_frames_to_gif(frames, self.out_path, fps=5)
        img = Image.open(self.out_path)
        # First frame should be red — check that R channel >> B channel
        first_frame = np.array(img.convert("RGB"))
        self.assertGreater(first_frame[0, 0, 0], 200)   # R channel high
        self.assertLess(first_frame[0, 0, 2], 50)        # B channel low


if __name__ == '__main__':
    unittest.main()
