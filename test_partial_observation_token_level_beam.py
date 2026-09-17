from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch

import benchmark_partial_observation_token_level_beam as planner


class TokenLevelBeamTests(unittest.TestCase):
    def test_planner_has_no_hidden_environment_arguments(self):
        parameters = set(inspect.signature(planner.plan_token_level_beam).parameters)
        forbidden = {"maze", "cell", "current", "goal", "simulator", "environment"}
        self.assertFalse(parameters & forbidden)

    def test_prunes_actions_before_predicting_wall_tokens(self):
        action_offset = 10
        wall_offset = 20
        goal_offset = 532
        delta_offset = 542
        delta_min = -11
        delta_count = 23
        vocabulary = delta_offset + delta_count
        action_deltas = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int16)
        history = [1, wall_offset, goal_offset, delta_offset + 13, delta_offset + 13]
        history_types = [1, 2, 3, 4, 4]
        calls = []

        def fake_logits(model, token_batches, type_batches, device):
            calls.append([list(tokens) for tokens in token_batches])
            output = torch.zeros((len(token_batches), vocabulary), dtype=torch.float32)
            stage = len(calls)
            if stage == 1:
                output[:, action_offset:action_offset + 4] = torch.tensor([4.0, 3.0, 2.0, 1.0])
            elif stage == 2:
                output[:, wall_offset:wall_offset + 512] = torch.arange(512, dtype=torch.float32)
            elif stage == 3:
                output[:, goal_offset:goal_offset + 10] = torch.arange(10, dtype=torch.float32)
            return output

        with patch.object(planner, "batched_logits", side_effect=fake_logits):
            action, _ = planner.plan_token_level_beam(
                object(),
                history,
                history_types,
                action_deltas,
                (action_offset, wall_offset, goal_offset, delta_offset, delta_min, delta_count),
                1,
                2,
                object(),
            )

        wall_stage_actions = [tokens[-1] - action_offset for tokens in calls[1]]
        self.assertEqual(wall_stage_actions, [0, 1])
        self.assertIn(action, (0, 1))

    def test_retains_best_prefixes_globally(self):
        candidates = [
            {"score": -2.0, "tokens": [2]},
            {"score": -1.0, "tokens": [3]},
            {"score": -1.0, "tokens": [1]},
        ]
        retained = planner.retain_best(candidates, 2)
        self.assertEqual([item["tokens"] for item in retained], [[1], [3]])


if __name__ == "__main__":
    unittest.main()
