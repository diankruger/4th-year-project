from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_partial_observation_cv_dataset import local_encoding


Direction = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic Grid-Bug2 on the partial-observation maze queries."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--queries-per-maze", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=19)
    parser.add_argument("--mazes", nargs="*", required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--per-maze-summary-csv", type=Path, required=True)
    return parser.parse_args()


def four_connected_m_line(goal_relative: Direction) -> list[Direction]:
    goal = np.asarray(goal_relative, dtype=np.int16)
    current = np.zeros(2, dtype=np.int16)
    line: list[Direction] = [(0, 0)]
    while not np.array_equal(current, goal):
        candidates: list[np.ndarray] = []
        if current[0] != goal[0]:
            step = current.copy()
            step[0] += 1 if goal[0] > current[0] else -1
            candidates.append(step)
        if current[1] != goal[1]:
            step = current.copy()
            step[1] += 1 if goal[1] > current[1] else -1
            candidates.append(step)

        def rank(cell: np.ndarray) -> tuple[int, int, int]:
            cross = abs(int(cell[0] * goal[1] - cell[1] * goal[0]))
            remaining = int(np.abs(goal - cell).sum())
            row_first = 0 if int(cell[0]) != int(current[0]) else 1
            return cross, remaining, row_first

        current = min(candidates, key=rank)
        line.append((int(current[0]), int(current[1])))
    return line


def rotate_right(direction: Direction) -> Direction:
    dr, dc = direction
    return dc, -dr


def rotate_left(direction: Direction) -> Direction:
    dr, dc = direction
    return -dc, dr


def reverse(direction: Direction) -> Direction:
    dr, dc = direction
    return -dr, -dc


def valid_directions_from_wall_mask(
    wall_mask: int, action_deltas: np.ndarray
) -> list[Direction]:
    valid: list[Direction] = []
    for dr_raw, dc_raw in action_deltas:
        dr, dc = int(dr_raw), int(dc_raw)
        bit = (dr + 1) * 3 + (dc + 1)
        if ((wall_mask >> bit) & 1) == 0:
            valid.append((dr, dc))
    return valid


class GridBug2:
    def __init__(self, goal_relative: Direction) -> None:
        self.line = four_connected_m_line(goal_relative)
        self.line_index = {cell: index for index, cell in enumerate(self.line)}
        self.goal = goal_relative
        self.current: Direction = (0, 0)
        self.mode = "goal"
        self.heading: Direction | None = None
        self.hit_index = -1
        self.hit_distance = len(self.line) - 1

    def _next_line_direction(self) -> Direction | None:
        index = self.line_index.get(self.current)
        if index is None or index + 1 >= len(self.line):
            return None
        nxt = self.line[index + 1]
        return nxt[0] - self.current[0], nxt[1] - self.current[1]

    def _boundary_direction(self, valid: set[Direction]) -> Direction:
        if self.heading is None:
            raise RuntimeError("Boundary following requires a heading")
        order = (
            rotate_right(self.heading),
            self.heading,
            rotate_left(self.heading),
            reverse(self.heading),
        )
        for direction in order:
            if direction in valid:
                return direction
        raise RuntimeError("No valid adjacent move in local observation")

    def choose(self, wall_mask: int, action_deltas: np.ndarray) -> Direction:
        valid = set(valid_directions_from_wall_mask(wall_mask, action_deltas))
        if not valid:
            raise RuntimeError("No valid adjacent move in local observation")

        preferred = self._next_line_direction()
        if self.mode == "goal":
            if preferred is not None and preferred in valid:
                chosen = preferred
            else:
                self.mode = "boundary"
                self.hit_index = self.line_index.get(self.current, -1)
                self.hit_distance = abs(self.goal[0] - self.current[0]) + abs(
                    self.goal[1] - self.current[1]
                )
                self.heading = preferred
                if self.heading is None:
                    raise RuntimeError("Goal-directed move is undefined before reaching goal")
                chosen = self._boundary_direction(valid)
        else:
            index = self.line_index.get(self.current)
            distance = abs(self.goal[0] - self.current[0]) + abs(
                self.goal[1] - self.current[1]
            )
            can_leave = (
                index is not None
                and index > self.hit_index
                and distance < self.hit_distance
                and preferred is not None
                and preferred in valid
            )
            if can_leave:
                self.mode = "goal"
                chosen = preferred
            else:
                chosen = self._boundary_direction(valid)

        self.heading = chosen
        self.current = self.current[0] + chosen[0], self.current[1] + chosen[1]
        return chosen


def rollout(
    maze: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    action_deltas: np.ndarray,
    max_steps: int,
) -> tuple[np.ndarray, int, int, int]:
    current = start.astype(np.int16).copy()
    goal_relative = int(goal[0] - start[0]), int(goal[1] - start[1])
    policy = GridBug2(goal_relative)
    delta_to_action = {
        (int(delta[0]), int(delta[1])): index
        for index, delta in enumerate(action_deltas)
    }
    path = [current.copy()]
    boundary_steps = 0
    mode_switches = 0
    invalid = 0

    for _ in range(max_steps):
        previous_mode = policy.mode
        wall_mask, _, _ = local_encoding(maze, current, goal)
        direction = policy.choose(wall_mask, action_deltas)
        if policy.mode == "boundary":
            boundary_steps += 1
        if policy.mode != previous_mode:
            mode_switches += 1
        action = delta_to_action[direction]
        proposed = current + action_deltas[action]
        inside = (
            0 <= int(proposed[0]) < maze.shape[0]
            and 0 <= int(proposed[1]) < maze.shape[1]
        )
        if not inside or int(maze[int(proposed[0]), int(proposed[1])]) != 0:
            invalid += 1
        else:
            current = proposed.astype(np.int16)
        path.append(current.copy())
        if np.array_equal(current, goal):
            break
    return np.asarray(path), invalid, boundary_steps, mode_switches


def summarize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        successful = group.loc[group["goal_reached"], "steps_until_stop"]
        row = dict(zip(columns, key_values))
        row.update(
            completion_rate=float(group["goal_reached"].mean()),
            average_steps_to_goal=(
                float(successful.mean()) if len(successful) else float("nan")
            ),
            average_compute_seconds=float(group["compute_seconds"].mean()),
            average_seconds_per_step=float(group["seconds_per_step"].mean()),
            invalid_real_action_count=int(group["invalid_real_action_count"].sum()),
            invalid_real_action_rate=float(group["invalid_real_action_rate"].mean()),
            average_boundary_steps=float(group["boundary_steps"].mean()),
            average_mode_switches=float(group["mode_switches"].mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    with np.load(args.dataset, allow_pickle=True) as data:
        names = np.asarray(data["maze_name_per_episode"]).astype(str)
        folds = np.asarray(data["fold_ids"])
        starts = np.asarray(data["start_cells"])
        goals = np.asarray(data["goal_cells"])
        lengths = np.asarray(data["path_lengths"])
        maze_indices = np.asarray(data["maze_indices"])
        episode_ids = np.asarray(data["episode_ids"])
        mazes = [np.asarray(item, dtype=np.int8) for item in data["mazes"]]
        action_deltas = np.asarray(data["action_deltas"], dtype=np.int16)

    rows: list[dict[str, object]] = []
    for maze_name in args.mazes:
        indices = np.flatnonzero((folds == args.cv_fold) & (names == maze_name))
        order = np.lexsort(
            (
                goals[indices, 1],
                goals[indices, 0],
                starts[indices, 1],
                starts[indices, 0],
                lengths[indices],
            )
        )
        for index in indices[order][: args.queries_per_maze]:
            maze = mazes[int(maze_indices[index])]
            start = starts[index].astype(np.int16)
            goal = goals[index].astype(np.int16)
            before = time.perf_counter()
            path, invalid, boundary_steps, mode_switches = rollout(
                maze, start, goal, action_deltas, args.max_steps
            )
            elapsed = time.perf_counter() - before
            steps = len(path) - 1
            rows.append(
                {
                    "cv_fold": args.cv_fold,
                    "maze_name": maze_name,
                    "episode_id": int(episode_ids[index]),
                    "query_index": int(index),
                    "strategy": "grid_bug2",
                    "steps_until_stop": steps,
                    "compute_seconds": elapsed,
                    "seconds_per_step": elapsed / max(steps, 1),
                    "goal_reached": bool(np.array_equal(path[-1], goal)),
                    "invalid_real_action_count": invalid,
                    "invalid_real_action_rate": invalid / max(steps, 1),
                    "boundary_steps": boundary_steps,
                    "mode_switches": mode_switches,
                    "max_steps": args.max_steps,
                }
            )

    results = pd.DataFrame(rows)
    summary = summarize(results, ["strategy"]).sort_values("strategy")
    per_maze = summarize(results, ["maze_name", "strategy"]).sort_values(
        ["maze_name", "strategy"]
    )
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    per_maze.to_csv(args.per_maze_summary_csv, index=False)
    print(json.dumps({"fold": args.cv_fold, "rows": len(results)}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
