from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

import benchmark_partial_observation_joint_receding_cv_decoding as planner
from build_partial_observation_joint_cv_dataset import main as build_dataset


class NoOracleTests(unittest.TestCase):
    def test_auto_device_falls_back_to_cpu_without_cuda(self):
        from unittest.mock import patch
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(str(planner.choose_device("auto")), "cpu")

    def test_planner_interface_has_no_hidden_environment_arguments(self):
        parameters = set(inspect.signature(planner.plan_no_oracle_batched).parameters)
        forbidden = {"maze", "cell", "current", "goal", "simulator", "environment"}
        self.assertFalse(parameters & forbidden)

    def test_action_mask_uses_wall_observation_only(self):
        deltas = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]])
        north_bit = 1 << 1
        self.assertNotIn(0, planner.observed_action_indices(north_bit, deltas))
        self.assertEqual(planner.observed_action_indices(0, deltas), [0, 1, 2, 3])

    def test_context_trim_keeps_complete_groups(self):
        tokens, types = list(range(40)), list(range(40))
        trimmed, trimmed_types = planner.trim_complete_groups(tokens, types, 20)
        self.assertEqual(len(trimmed), 20)
        self.assertEqual((len(trimmed) - 5) % 5, 0)
        self.assertEqual(trimmed[:5], tokens[:5])
        self.assertEqual(trimmed_types[:5], types[:5])

    def test_imagined_planner_source_does_not_call_real_observation_generator(self):
        source = inspect.getsource(planner.plan_no_oracle_batched)
        self.assertNotIn("observation_tokens(", source)
        self.assertNotIn("local_encoding(", source)

    def test_permanent_history_receives_real_not_predicted_observation(self):
        source = inspect.getsource(planner.rollout)
        self.assertIn("history.extend([action_offset + action] + real_obs)", source)
        self.assertNotIn("history.extend([action_offset + action] + predicted)", source)


class DatasetIntegrityTests(unittest.TestCase):
    def test_controlled_metadata_and_target_counts(self):
        source_path = Path("maze2d_all_mazes_transformer_ready_4act_cv5.npz")
        if not source_path.exists():
            self.skipTest("source dataset not present")
        import sys
        old = sys.argv
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "joint.npz"
            try:
                sys.argv = ["builder", "--source", str(source_path), "--output", str(output)]
                build_dataset()
            finally:
                sys.argv = old
            with np.load(source_path, allow_pickle=True) as source, np.load(output, allow_pickle=True) as built:
                for key in ("episode_ids", "fold_ids", "start_cells", "goal_cells", "maze_indices", "path_lengths"):
                    self.assertTrue(np.array_equal(source[key], built[key]), key)
                action_count = int((built["action_labels"] != -100).sum())
                observation_count = int(built["supervised_observation_mask"].sum())
                self.assertEqual(observation_count, 4 * action_count)


if __name__ == "__main__":
    unittest.main()
