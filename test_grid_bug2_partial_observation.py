import unittest

import numpy as np

from evaluate_grid_bug2_partial_observation import (
    GridBug2,
    four_connected_m_line,
    rotate_left,
    rotate_right,
    valid_directions_from_wall_mask,
)


DELTAS = np.asarray([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.int16)


class GridBug2Tests(unittest.TestCase):
    def test_m_line_is_four_connected_and_reaches_goal(self):
        line = four_connected_m_line((4, 7))
        self.assertEqual(line[0], (0, 0))
        self.assertEqual(line[-1], (4, 7))
        self.assertEqual(len(line), 12)
        for first, second in zip(line, line[1:]):
            self.assertEqual(abs(first[0] - second[0]) + abs(first[1] - second[1]), 1)

    def test_wall_mask_exposes_only_locally_free_actions(self):
        north_bit = 1 << 1
        east_bit = 1 << 5
        valid = valid_directions_from_wall_mask(north_bit | east_bit, DELTAS)
        self.assertEqual(valid, [(0, -1), (1, 0)])

    def test_right_hand_order_turns_right_when_goal_move_is_blocked(self):
        policy = GridBug2((0, 3))
        east_bit = 1 << 5
        chosen = policy.choose(east_bit, DELTAS)
        self.assertEqual(chosen, (1, 0))
        self.assertEqual(policy.mode, "boundary")

    def test_rotation_matches_grid_orientation(self):
        self.assertEqual(rotate_right((-1, 0)), (0, 1))
        self.assertEqual(rotate_left((-1, 0)), (0, -1))


if __name__ == "__main__":
    unittest.main()
