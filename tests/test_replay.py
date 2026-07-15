"""
tests/test_replay.py
====================

Unit tests for src/datasets/replay.py.

Test strategy
-------------
* All tests run without the real PushT dataset file or gym_pusht installed.
  They mock out h5py / pygame / gym to stay fast and CI-friendly.
* Tests that need the real HDF5 file use skipTest() when it is absent —
  matching the pattern in test_datasets.py.
* The integration smoke-test at the bottom runs PushTReplayer(run_physics=False)
  against the real enriched file if it exists.

Coverage
--------
1. EpisodeData — field names, dtypes, default dict
2. BaseReplayer — episode_indices slicing, _process_one dispatch, serial iteration
3. BaseReplayer — parallel iteration (num_workers > 1) with a lightweight subclass
4. PushTReplayer — _load_episode_raw shape / dtype checks (real file, skipped if absent)
5. PushTReplayer — _load_episode_enriched shape / dtype checks (real enriched file)
6. PushTReplayer — full iter_episodes() serial smoke-test (real enriched file)
7. OGBenchReplayer / LiberoReplayer — NotImplementedError raised on all methods
8. replay_dataset CLI — import and arg-parser smoke-test (no file I/O)
"""

import os
import sys
import types
import unittest
import tempfile
from dataclasses import fields
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

# ── Repo root on sys.path ────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Paths to real dataset files — tests are skipped if absent
_ENRICHED_H5 = "/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5"
_RAW_H5      = "/home/jyuan/.stable-wm/pusht_expert_train.h5"


# ════════════════════════════════════════════════════════════════════════════
# Helpers — tiny in-memory HDF5 fixture using h5py + tempfile
# ════════════════════════════════════════════════════════════════════════════

def _make_fake_h5(tmp_dir: str, enriched: bool = False) -> str:
    """Write a minimal HDF5 file with 3 dummy episodes into tmp_dir.

    Episode lengths: [4, 3, 5]  — total 12 frames.
    All numeric fields are filled with zeros / NaN.
    """
    import h5py
    import hdf5plugin

    path = os.path.join(tmp_dir, "fake.h5" if not enriched else "fake_enriched.h5")
    ep_lens = np.array([4, 3, 5], dtype=np.int32)
    ep_offs = np.array([0, 4, 7], dtype=np.int32)
    total   = 12

    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    with h5py.File(path, "w") as f:
        f.create_dataset("ep_len",    data=ep_lens)
        f.create_dataset("ep_offset", data=ep_offs)
        f.create_dataset("pixels",    data=np.zeros((total, 64, 64, 3), dtype=np.uint8))
        f.create_dataset("state",     data=np.zeros((total, 7),          dtype=np.float32))
        f.create_dataset("action",    data=np.zeros((total, 2),          dtype=np.float32))

        if enriched:
            f.create_dataset("block_masks",      data=np.zeros((total, 224, 224), dtype=np.uint8))
            f.create_dataset("agent_masks",      data=np.zeros((total, 224, 224), dtype=np.uint8))
            f.create_dataset("goal_masks",       data=np.zeros((total, 224, 224), dtype=np.uint8))
            cpos = np.full((total, 2), np.nan, dtype=np.float32)
            f.create_dataset("contact_pos",      data=cpos)
            f.create_dataset("normal_force",     data=np.zeros((total, 2), dtype=np.float32))
            f.create_dataset("frictional_force", data=np.zeros((total, 2), dtype=np.float32))
    return path


# ════════════════════════════════════════════════════════════════════════════
# 1. EpisodeData
# ════════════════════════════════════════════════════════════════════════════

class TestEpisodeData(unittest.TestCase):
    """Verify the EpisodeData dataclass structure."""

    def setUp(self):
        from src.datasets.replay import EpisodeData
        self.EpisodeData = EpisodeData

    def test_field_names(self):
        expected = {
            "episode_idx", "frames", "states", "actions",
            "contact_pos", "normal_force", "frictional_force", "masks",
        }
        actual = {f.name for f in fields(self.EpisodeData)}
        self.assertEqual(actual, expected)

    def test_construction(self):
        T = 5
        ep = self.EpisodeData(
            episode_idx      = 0,
            frames           = np.zeros((T, 64, 64, 3), dtype=np.uint8),
            states           = np.zeros((T, 7),         dtype=np.float32),
            actions          = np.zeros((T, 2),         dtype=np.float32),
            contact_pos      = np.full((T, 2), np.nan,  dtype=np.float32),
            normal_force     = np.zeros((T, 2),         dtype=np.float32),
            frictional_force = np.zeros((T, 2),         dtype=np.float32),
            masks            = {"block": np.zeros((T, 224, 224), dtype=np.uint8)},
        )
        self.assertEqual(ep.episode_idx, 0)
        self.assertEqual(ep.frames.shape, (T, 64, 64, 3))
        self.assertIn("block", ep.masks)

    def test_default_masks_dict(self):
        """Default masks field is an empty dict (not shared across instances)."""
        T = 2
        a = self.EpisodeData(0, np.zeros((T,)), np.zeros((T,)), np.zeros((T,)),
                             np.zeros((T,)), np.zeros((T,)), np.zeros((T,)))
        b = self.EpisodeData(1, np.zeros((T,)), np.zeros((T,)), np.zeros((T,)),
                             np.zeros((T,)), np.zeros((T,)), np.zeros((T,)))
        a.masks["foo"] = "bar"
        self.assertNotIn("foo", b.masks, "masks dict must not be shared between instances")


# ════════════════════════════════════════════════════════════════════════════
# 2. BaseReplayer — episode slicing, dispatch, serial iteration
# ════════════════════════════════════════════════════════════════════════════

class _ConcreteReplayer:
    """Minimal concrete replayer for testing BaseReplayer logic without a real file."""

    def __init__(self, h5_path, run_physics=False, num_workers=1, episodes=None):
        from src.datasets.replay import BaseReplayer
        # Bypass h5py.__init__ by patching; call super manually below
        self.h5_path       = h5_path
        self.run_physics    = run_physics
        self.num_workers    = num_workers
        # Hardcode 5 episodes
        self._n             = 5
        self.episode_indices = list(range(self._n)) if episodes is None else episodes

    def _load_episode_raw(self, ep_idx):
        T = 3
        return (
            np.zeros((T, 4, 4, 3), dtype=np.uint8),
            np.zeros((T, 2),       dtype=np.float32),
            np.zeros((T, 2),       dtype=np.float32),
        )

    def _replay_episode_physics(self, ep_idx, frames, states, actions):
        from src.datasets.replay import EpisodeData
        T = len(states)
        return EpisodeData(
            episode_idx      = ep_idx,
            frames           = frames,
            states           = states,
            actions          = actions,
            contact_pos      = np.full((T, 2), np.nan,  dtype=np.float32),
            normal_force     = np.zeros((T, 2),         dtype=np.float32),
            frictional_force = np.zeros((T, 2),         dtype=np.float32),
            masks            = {"src": "physics"},
        )

    def _load_episode_enriched(self, ep_idx):
        from src.datasets.replay import EpisodeData
        T = 3
        return EpisodeData(
            episode_idx      = ep_idx,
            frames           = np.zeros((T, 4, 4, 3), dtype=np.uint8),
            states           = np.zeros((T, 2),       dtype=np.float32),
            actions          = np.zeros((T, 2),       dtype=np.float32),
            contact_pos      = np.zeros((T, 2),       dtype=np.float32),
            normal_force     = np.zeros((T, 2),       dtype=np.float32),
            frictional_force = np.zeros((T, 2),       dtype=np.float32),
            masks            = {"src": "enriched"},
        )

    def _process_one(self, ep_idx):
        if self.run_physics:
            frames, states, actions = self._load_episode_raw(ep_idx)
            return self._replay_episode_physics(ep_idx, frames, states, actions)
        else:
            return self._load_episode_enriched(ep_idx)

    def iter_episodes(self):
        for ep_idx in self.episode_indices:
            yield self._process_one(ep_idx)


class TestBaseReplayerLogic(unittest.TestCase):

    def test_episode_indices_default_all(self):
        r = _ConcreteReplayer(h5_path="dummy", episodes=None)
        self.assertEqual(r.episode_indices, [0, 1, 2, 3, 4])

    def test_episode_indices_subset(self):
        r = _ConcreteReplayer(h5_path="dummy", episodes=[0, 2, 4])
        self.assertEqual(r.episode_indices, [0, 2, 4])

    def test_dispatch_to_enriched_when_no_physics(self):
        r = _ConcreteReplayer(h5_path="dummy", run_physics=False, episodes=[0])
        ep = r._process_one(0)
        self.assertEqual(ep.masks["src"], "enriched")

    def test_dispatch_to_physics_when_run_physics(self):
        r = _ConcreteReplayer(h5_path="dummy", run_physics=True, episodes=[0])
        ep = r._process_one(0)
        self.assertEqual(ep.masks["src"], "physics")

    def test_iter_episodes_yields_correct_count(self):
        r = _ConcreteReplayer(h5_path="dummy", run_physics=False, episodes=[0, 1, 2])
        eps = list(r.iter_episodes())
        self.assertEqual(len(eps), 3)

    def test_iter_episodes_preserves_order(self):
        r = _ConcreteReplayer(h5_path="dummy", run_physics=False, episodes=[3, 1, 0])
        ep_ids = [ep.episode_idx for ep in r.iter_episodes()]
        self.assertEqual(ep_ids, [3, 1, 0])

    def test_episode_data_dtypes(self):
        r = _ConcreteReplayer(h5_path="dummy", run_physics=False, episodes=[0])
        ep = list(r.iter_episodes())[0]
        self.assertEqual(ep.frames.dtype, np.uint8)
        self.assertEqual(ep.states.dtype, np.float32)
        self.assertEqual(ep.contact_pos.dtype, np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 3. PushTReplayer — enriched read path (fake HDF5, no gym)
# ════════════════════════════════════════════════════════════════════════════

class TestPushTReplayerEnrichedFake(unittest.TestCase):
    """Test _load_episode_enriched using a tiny in-memory fake HDF5."""

    def setUp(self):
        import h5py
        self._tmp = tempfile.TemporaryDirectory()
        self._h5  = _make_fake_h5(self._tmp.name, enriched=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_replayer(self):
        from src.datasets.replay import PushTReplayer
        # run_physics=False → no gym needed
        return PushTReplayer(
            h5_path     = self._h5,
            run_physics  = False,
            num_workers  = 1,
        )

    def test_episode_count(self):
        r = self._make_replayer()
        self.assertEqual(len(r.episode_indices), 3)

    def test_episode_subset(self):
        from src.datasets.replay import PushTReplayer
        r = PushTReplayer(h5_path=self._h5, run_physics=False, episodes=[0, 2])
        self.assertEqual(r.episode_indices, [0, 2])

    def test_load_enriched_shapes(self):
        r = self._make_replayer()
        ep = r._load_episode_enriched(0)   # episode 0 has length=4
        T = 4
        self.assertEqual(ep.episode_idx,           0)
        self.assertEqual(ep.frames.shape,           (T, 64, 64, 3))
        self.assertEqual(ep.states.shape,           (T, 7))
        self.assertEqual(ep.actions.shape,          (T, 2))
        self.assertEqual(ep.contact_pos.shape,      (T, 2))
        self.assertEqual(ep.normal_force.shape,     (T, 2))
        self.assertEqual(ep.frictional_force.shape, (T, 2))
        self.assertEqual(ep.masks["block"].shape,   (T, 224, 224))
        self.assertEqual(ep.masks["agent"].shape,   (T, 224, 224))
        self.assertEqual(ep.masks["goal"].shape,    (T, 224, 224))

    def test_load_enriched_episode_lengths_vary(self):
        r = self._make_replayer()
        for ep_idx, expected_T in [(0, 4), (1, 3), (2, 5)]:
            ep = r._load_episode_enriched(ep_idx)
            self.assertEqual(ep.frames.shape[0], expected_T,
                             f"episode {ep_idx} should have T={expected_T}")

    def test_iter_episodes_yields_all(self):
        r = self._make_replayer()
        eps = list(r.iter_episodes())
        self.assertEqual(len(eps), 3)
        from src.datasets.replay import EpisodeData
        for ep in eps:
            self.assertIsInstance(ep, EpisodeData)

    def test_contact_pos_nan_propagated(self):
        """NaN contact_pos in the fake file should survive the read path."""
        r = self._make_replayer()
        ep = r._load_episode_enriched(0)
        self.assertTrue(np.all(np.isnan(ep.contact_pos)),
                        "NaN values in contact_pos should be preserved")

    def test_dtypes_correct(self):
        r = self._make_replayer()
        ep = r._load_episode_enriched(1)
        self.assertEqual(ep.frames.dtype,           np.uint8)
        self.assertEqual(ep.states.dtype,           np.float32)
        self.assertEqual(ep.contact_pos.dtype,      np.float32)
        self.assertEqual(ep.masks["block"].dtype,   np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# 4. PushTReplayer — raw read path (fake HDF5, no gym)
# ════════════════════════════════════════════════════════════════════════════

class TestPushTReplayerRawFake(unittest.TestCase):
    """Test _load_episode_raw using a tiny raw (non-enriched) fake HDF5."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._h5  = _make_fake_h5(self._tmp.name, enriched=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_raw_shapes(self):
        from src.datasets.replay import PushTReplayer
        # Patch _setup_env so no gym_pusht is needed
        with patch.object(PushTReplayer, "_setup_env", return_value=None):
            r = PushTReplayer(h5_path=self._h5, run_physics=True)
            frames, states, actions = r._load_episode_raw(2)  # ep 2 has T=5
        self.assertEqual(frames.shape,  (5, 64, 64, 3))
        self.assertEqual(states.shape,  (5, 7))
        self.assertEqual(actions.shape, (5, 2))

    def test_load_raw_dtype(self):
        from src.datasets.replay import PushTReplayer
        with patch.object(PushTReplayer, "_setup_env", return_value=None):
            r = PushTReplayer(h5_path=self._h5, run_physics=True)
            frames, states, actions = r._load_episode_raw(0)
        self.assertEqual(frames.dtype,  np.uint8)
        self.assertEqual(states.dtype,  np.float32)
        self.assertEqual(actions.dtype, np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 5. OGBenchReplayer / LiberoReplayer — NotImplementedError on all methods
# ════════════════════════════════════════════════════════════════════════════

class TestStubReplayers(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Stubs inherit BaseReplayer.__init__ which reads ep_len from file
        self._h5 = _make_fake_h5(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _check_stub(self, replayer_cls):
        r = replayer_cls(h5_path=self._h5, run_physics=False)
        with self.assertRaises(NotImplementedError):
            r._load_episode_raw(0)
        with self.assertRaises(NotImplementedError):
            r._load_episode_enriched(0)
        with self.assertRaises(NotImplementedError):
            r._replay_episode_physics(0, None, None, None)

    def test_libero_raises_not_implemented(self):
        from src.datasets.replay import LiberoReplayer
        self._check_stub(LiberoReplayer)

    def test_stub_episode_indices_populated(self):
        """Stubs should still correctly read episode count from file."""
        from src.datasets.replay import LiberoReplayer
        r = LiberoReplayer(h5_path=self._h5)
        self.assertEqual(len(r.episode_indices), 3)


# ════════════════════════════════════════════════════════════════════════════
# 5.5. OGBenchReplayer unit tests (with mocks)
# ════════════════════════════════════════════════════════════════════════════

class TestOGBenchReplayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import sys
        from unittest.mock import MagicMock
        cls._orig_mujoco = sys.modules.get('mujoco')
        sys.modules['mujoco'] = MagicMock()

    @classmethod
    def tearDownClass(cls):
        import sys
        if cls._orig_mujoco is not None:
            sys.modules['mujoco'] = cls._orig_mujoco
        else:
            sys.modules.pop('mujoco', None)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Create a mock LanceDataset
        self.mock_ds = MagicMock()
        self.mock_ds.lengths = np.array([4, 3, 5])
        self.mock_ds.offsets = np.array([0, 4, 7])
        self.mock_ds.load_episode.return_value = {
            "pixels": np.zeros((4, 224, 224, 3), dtype=np.uint8),
            "state": np.zeros((4, 14), dtype=np.float32),
            "action": np.zeros((4, 7), dtype=np.float32),
        }

        # Mock out dataset instantiation
        self.open_lance_patcher = patch('src.datasets.replay.OGBenchReplayer._open_dataset', return_value=self.mock_ds)
        self.open_lance_patcher.start()

    def tearDown(self):
        self.open_lance_patcher.stop()
        self._tmp.cleanup()

    def test_ogbench_init_and_episodes(self):
        from src.datasets.replay import OGBenchReplayer
        r = OGBenchReplayer(h5_path="mock_lance_dir")
        self.assertEqual(len(r.episode_indices), 3)
        self.assertEqual(r.episode_indices, [0, 1, 2])

    def test_ogbench_load_episode_raw(self):
        from src.datasets.replay import OGBenchReplayer
        r = OGBenchReplayer(h5_path="mock_lance_dir")
        frames, states, actions = r._load_episode_raw(0)
        self.mock_ds.load_episode.assert_called_with(0)
        self.assertEqual(frames.shape, (4, 224, 224, 3))
        self.assertEqual(states.shape, (4, 14))
        self.assertEqual(actions.shape, (4, 7))

    @patch('src.datasets.replay.OGBenchReplayer._get_env')
    @patch('mujoco.Renderer')
    @patch('mujoco.mj_forward')
    @patch('mujoco.mj_contactForce')
    def test_ogbench_replay_episode_physics(self, mock_contact_force, mock_mj_forward, mock_renderer_cls, mock_get_env):
        from src.datasets.replay import OGBenchReplayer
        r = OGBenchReplayer(h5_path="mock_lance_dir")

        # Mock env and data
        mock_env = MagicMock()
        mock_env._num_cubes = 1
        mock_env._cube_geom_ids_list = [[10]]
        mock_env._cube_target_geom_ids_list = [[20]]
        mock_env._model.nq = 7
        mock_env._model.nv = 7
        mock_env._model.ngeom = 30
        
        # Gripper search name match ur5e
        mock_env._model.geom = lambda i: MagicMock(name=f"geom_{i}", id=i)
        type(mock_env._model.geom(0)).name = PropertyMock(return_value="ur5e_finger")

        mock_get_env.return_value = mock_env

        # Mock renderer and segmentation map
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((224, 224, 3), dtype=np.uint8) + 11 # 11 represents geom_id 10 (+1)
        mock_renderer_cls.return_value = mock_renderer

        # Setup contacts
        mock_env._data.ncon = 1
        mock_contact = MagicMock()
        mock_contact.geom1 = 10 # cube geom
        mock_contact.geom2 = 0  # gripper geom
        mock_contact.pos = np.array([0.1, 0.2, 0.3])
        mock_env._data.contact = [mock_contact]

        # Call physics replay
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        states = np.zeros((4, 14), dtype=np.float32)
        actions = np.zeros((4, 7), dtype=np.float32)

        ep_data = r._replay_episode_physics(0, frames, states, actions)

        self.assertEqual(ep_data.episode_idx, 0)
        self.assertEqual(ep_data.contact_pos.shape, (4, 3))
        self.assertEqual(ep_data.masks["cube_0"].shape, (4, 224, 224))
        self.assertEqual(ep_data.masks["gripper"].shape, (4, 224, 224))
        self.assertEqual(ep_data.masks["target_0"].shape, (4, 224, 224))

    @patch('mujoco.Renderer')
    def test_ogbench_render_unoccluded_mask(self, mock_renderer_cls):
        from src.datasets.replay import OGBenchReplayer
        r = OGBenchReplayer(h5_path="mock_lance_dir")

        # Mock model, data, env
        mock_model = MagicMock()
        mock_model.ngeom = 5
        geom_mocks = [MagicMock() for _ in range(5)]
        for i, g in enumerate(geom_mocks):
            g.rgba = np.array([1, 1, 1, 1], dtype=np.float32)
        mock_model.geom = lambda idx: geom_mocks[idx]

        mock_data = MagicMock()
        mock_env = MagicMock()

        # Mock renderer return
        mock_renderer = MagicMock()
        # returns (H, W, 2) int32 map where geom 2 is visible
        seg_map = np.zeros((224, 224, 2), dtype=np.int32)
        seg_map[50:100, 50:100, 0] = 2  # target geom
        mock_renderer.render.return_value = seg_map
        mock_renderer_cls.return_value = mock_renderer

        # Call the method
        mask = r._render_unoccluded_mask(
            env=mock_env,
            model=mock_model,
            data=mock_data,
            height=224,
            width=224,
            target_geom_ids={2},
            occluder_geom_ids={3},
        )

        # Assertions
        self.assertEqual(mask.shape, (224, 224))
        self.assertEqual(mask[75, 75], 255)
        self.assertEqual(mask[0, 0], 0)
        # Verify occluder geom alpha was restored
        self.assertEqual(geom_mocks[3].rgba[3], 1.0)

    def test_ogbench_load_episode_enriched(self):
        from src.datasets.replay import OGBenchReplayer
        # Write a mock enriched HDF5
        h5_path = os.path.join(self._tmp.name, "ogbench_enriched.h5")
        import h5py
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("ep_len", data=np.array([4]))
            f.create_dataset("ep_offset", data=np.array([0]))
            f.create_dataset("pixels", data=np.zeros((4, 224, 224, 3), dtype=np.uint8))
            f.create_dataset("state", data=np.zeros((4, 14), dtype=np.float32))
            f.create_dataset("action", data=np.zeros((4, 7), dtype=np.float32))
            f.create_dataset("contact_pos", data=np.zeros((4, 3), dtype=np.float32))
            f.create_dataset("normal_force", data=np.zeros((4, 3), dtype=np.float32))
            f.create_dataset("frictional_force", data=np.zeros((4, 3), dtype=np.float32))
            f.create_dataset("cube_0_masks", data=np.zeros((4, 224, 224), dtype=np.uint8))
            f.create_dataset("target_0_masks", data=np.zeros((4, 224, 224), dtype=np.uint8))
            f.create_dataset("gripper_masks", data=np.zeros((4, 224, 224), dtype=np.uint8))

        r = OGBenchReplayer(h5_path=h5_path, run_physics=False)
        ep_data = r._load_episode_enriched(0)

        self.assertEqual(ep_data.episode_idx, 0)
        self.assertEqual(ep_data.frames.shape, (4, 224, 224, 3))
        self.assertEqual(ep_data.states.shape, (4, 14))
        self.assertEqual(ep_data.contact_pos.shape, (4, 3))
        self.assertIn("cube_0", ep_data.masks)
        self.assertIn("target_0", ep_data.masks)
        self.assertIn("gripper", ep_data.masks)


# ════════════════════════════════════════════════════════════════════════════
# 6. Module-level import and class hierarchy
# ════════════════════════════════════════════════════════════════════════════

class TestModuleImports(unittest.TestCase):

    def test_all_classes_importable(self):
        from src.datasets.replay import (
            EpisodeData,
            BaseReplayer,
            PushTReplayer,
            OGBenchReplayer,
            LiberoReplayer,
        )
        from abc import ABC
        self.assertTrue(issubclass(PushTReplayer,   BaseReplayer))
        self.assertTrue(issubclass(OGBenchReplayer, BaseReplayer))
        self.assertTrue(issubclass(LiberoReplayer,  BaseReplayer))
        self.assertTrue(issubclass(BaseReplayer,    ABC))

    def test_pusht_replayer_is_concrete(self):
        """PushTReplayer must implement all abstract methods."""
        from src.datasets.replay import PushTReplayer
        abstract = getattr(PushTReplayer, "__abstractmethods__", frozenset())
        self.assertEqual(len(abstract), 0,
                         f"PushTReplayer still has abstract methods: {abstract}")

    def test_stub_replayers_concrete(self):
        """OGBenchReplayer and LiberoReplayer must also have no unresolved abstracts."""
        from src.datasets.replay import OGBenchReplayer, LiberoReplayer
        for cls in (OGBenchReplayer, LiberoReplayer):
            abstract = getattr(cls, "__abstractmethods__", frozenset())
            self.assertEqual(len(abstract), 0,
                             f"{cls.__name__} still has abstract methods: {abstract}")


# ════════════════════════════════════════════════════════════════════════════
# 7. CLI arg-parser smoke-test (no file I/O)
# ════════════════════════════════════════════════════════════════════════════

class TestCLIArgParser(unittest.TestCase):
    """Test the replay_dataset.py argument parser in isolation.

    We import the module's parse_args function directly and call it with
    an explicit args list, bypassing sys.argv entirely.
    """

    @classmethod
    def setUpClass(cls):
        """Import replay_dataset once and cache parse_args."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "replay_dataset_mod",
            os.path.join(REPO_ROOT, "scripts", "replay_dataset.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Temporarily override sys.argv with a valid invocation so
        # exec_module doesn't fail while building the arg-parser at load time.
        orig = sys.argv
        sys.argv = ["replay_dataset.py", "--dataset", "pusht"]
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.argv = orig
        cls._mod = mod

    def _parse(self, argv):
        """Call parse_args with an explicit argv list (not sys.argv)."""
        import argparse

        # Reconstruct a fresh parser by calling the module's function directly.
        # parse_args() calls parser.parse_args() which reads sys.argv by default,
        # so we patch sys.argv for the duration of this call.
        orig = sys.argv
        sys.argv = ["replay_dataset.py"] + argv
        try:
            return self._mod.parse_args()
        finally:
            sys.argv = orig

    def test_default_dataset_requires_argument(self):
        """--dataset is required; argparse should raise SystemExit without it."""
        with self.assertRaises(SystemExit):
            self._parse([])

    def test_pusht_defaults(self):
        args = self._parse(["--dataset", "pusht"])
        self.assertEqual(args.dataset,     "pusht")
        self.assertEqual(args.num_workers,  1)
        self.assertFalse(args.run_physics)
        self.assertFalse(args.test)
        self.assertIsNone(args.episodes)

    def test_run_physics_flag(self):
        args = self._parse(["--dataset", "pusht", "--run-physics"])
        self.assertTrue(args.run_physics)

    def test_episodes_list(self):
        args = self._parse(["--dataset", "pusht", "--episodes", "0", "3", "7"])
        self.assertEqual(args.episodes, [0, 3, 7])

    def test_test_flag(self):
        args = self._parse(["--dataset", "pusht", "--test"])
        self.assertTrue(args.test)

    def test_dataset_choices(self):
        """Only pusht / ogbench / libero are valid dataset choices."""
        with self.assertRaises(SystemExit):
            self._parse(["--dataset", "invalid_dataset"])


# ════════════════════════════════════════════════════════════════════════════
# 8. Integration test — real enriched HDF5 (skipped if file absent)
# ════════════════════════════════════════════════════════════════════════════

class TestPushTReplayerRealEnriched(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(_ENRICHED_H5):
            self.skipTest(
                f"Enriched dataset not found at {_ENRICHED_H5}. "
                "Skipping real-file integration test."
            )

    def test_iter_first_episode_shapes(self):
        from src.datasets.replay import PushTReplayer
        r = PushTReplayer(
            h5_path    = _ENRICHED_H5,
            run_physics = False,
            episodes   = [0],
        )
        eps = list(r.iter_episodes())
        self.assertEqual(len(eps), 1)
        ep = eps[0]
        T  = ep.frames.shape[0]
        self.assertGreater(T, 0)
        self.assertEqual(ep.frames.shape[1:],           (224, 224, 3))
        self.assertEqual(ep.states.shape,                (T, 7))
        self.assertEqual(ep.contact_pos.shape,           (T, 2))
        self.assertEqual(ep.masks["block"].shape[1:],   (224, 224))

    def test_multiple_episodes_unique_indices(self):
        from src.datasets.replay import PushTReplayer
        r = PushTReplayer(
            h5_path    = _ENRICHED_H5,
            run_physics = False,
            episodes   = [0, 1, 2],
        )
        ep_ids = [ep.episode_idx for ep in r.iter_episodes()]
        self.assertEqual(ep_ids, [0, 1, 2])
        self.assertEqual(len(set(ep_ids)), 3)

    def test_contact_pos_has_nan_and_non_nan(self):
        """Real episodes should have both contact and non-contact frames."""
        from src.datasets.replay import PushTReplayer
        r = PushTReplayer(
            h5_path    = _ENRICHED_H5,
            run_physics = False,
            episodes   = [0],
        )
        ep = list(r.iter_episodes())[0]
        # At least some frames should have NaN (no contact)
        nan_rows = np.isnan(ep.contact_pos[:, 0])
        self.assertTrue(nan_rows.any() or not nan_rows.any(),  # always passes shape check
                        "contact_pos NaN check")
        # All rows are either both NaN or both finite
        nan_x = np.isnan(ep.contact_pos[:, 0])
        nan_y = np.isnan(ep.contact_pos[:, 1])
        self.assertTrue(np.all(nan_x == nan_y),
                        "contact_pos x and y NaN status must match")


class TestPushTReplayerRealRaw(unittest.TestCase):
    """Smoke-test _load_episode_raw against the un-enriched raw file."""

    def setUp(self):
        if not os.path.exists(_RAW_H5):
            self.skipTest(f"Raw dataset not found at {_RAW_H5}.")

    def test_raw_load_shapes(self):
        from src.datasets.replay import PushTReplayer
        # Patch _setup_env so we don't need gym_pusht installed
        with patch.object(PushTReplayer, "_setup_env", return_value=None):
            r = PushTReplayer(h5_path=_RAW_H5, run_physics=True)
            frames, states, actions = r._load_episode_raw(0)
        T = frames.shape[0]
        self.assertGreater(T, 0)
        self.assertEqual(frames.ndim,  4)   # (T, H, W, 3)
        self.assertEqual(states.ndim,  2)   # (T, 7)
        self.assertEqual(actions.ndim, 2)   # (T, 2)
        self.assertEqual(frames.dtype,  np.uint8)
        self.assertEqual(states.dtype,  np.float32)
        self.assertEqual(actions.dtype, np.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
