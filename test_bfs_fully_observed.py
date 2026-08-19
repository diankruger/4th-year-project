import unittest

import numpy as np

from evaluate_bfs_fully_observed import bfs_shortest_path


DELTAS = np.asarray([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.int16)


class BreadthFirstSearchTests(unittest.TestCase):
    def test_open_grid_returns_manhattan_shortest_path(self):
        maze = np.zeros((4, 5), dtype=np.int8)
        path, expanded, maximum_frontier = bfs_shortest_path(
            maze, (0, 0), (3, 4), DELTAS
        )
        self.assertEqual(len(path) - 1, 7)
        self.assertGreater(expanded, 0)
        self.assertGreaterEqual(maximum_frontier, 1)

    def test_obstacle_requires_optimal_detour(self):
        maze = np.zeros((5, 5), dtype=np.int8)
        maze[0:4, 2] = 1
        path, _, _ = bfs_shortest_path(maze, (0, 0), (0, 4), DELTAS)
        self.assertEqual(len(path) - 1, 12)
        self.assertTrue(all(maze[row, column] == 0 for row, column in path))

    def test_tie_breaking_is_deterministic_from_action_order(self):
        maze = np.zeros((3, 3), dtype=np.int8)
        first, _, _ = bfs_shortest_path(maze, (1, 1), (0, 2), DELTAS)
        second, _, _ = bfs_shortest_path(maze, (1, 1), (0, 2), DELTAS)
        self.assertEqual(first, second)
        self.assertEqual(first[1], (1, 2))

    def test_unreachable_goal_raises(self):
        maze = np.zeros((3, 3), dtype=np.int8)
        maze[1, :] = 1
        with self.assertRaises(RuntimeError):
            bfs_shortest_path(maze, (0, 0), (2, 2), DELTAS)


if __name__ == "__main__":
    unittest.main()
