from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic BFS on fully observed maze queries."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--per-maze-summary-csv", type=Path, required=True)
    parser.add_argument("--per-fold-summary-csv", type=Path, required=True)
    return parser.parse_args()


def bfs_shortest_path(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    action_deltas: np.ndarray,
) -> tuple[list[tuple[int, int]], int, int]:
    if start == goal:
        return [start], 0, 1

    frontier = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    expanded_states = 0
    maximum_frontier_size = 1

    while frontier:
        current = frontier.popleft()
        expanded_states += 1
        for dr_raw, dc_raw in action_deltas:
            neighbour = current[0] + int(dr_raw), current[1] + int(dc_raw)
            row, column = neighbour
            if not (0 <= row < maze.shape[0] and 0 <= column < maze.shape[1]):
                continue
            if int(maze[row, column]) != 0 or neighbour in parents:
                continue
            parents[neighbour] = current
            if neighbour == goal:
                path = [goal]
                cursor = goal
                while parents[cursor] is not None:
                    cursor = parents[cursor]
                    path.append(cursor)
                path.reverse()
                return path, expanded_states, maximum_frontier_size
            frontier.append(neighbour)
        maximum_frontier_size = max(maximum_frontier_size, len(frontier))

    raise RuntimeError(f"Goal {goal} is unreachable from {start}")


def summarize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns, key_values))
        row.update(
            queries=int(len(group)),
            completion_rate=float(group["goal_reached"].mean()),
            average_steps_to_goal=float(group["steps_to_goal"].mean()),
            minimum_steps_to_goal=int(group["steps_to_goal"].min()),
            maximum_steps_to_goal=int(group["steps_to_goal"].max()),
            average_compute_seconds=float(group["compute_seconds"].mean()),
            median_compute_seconds=float(group["compute_seconds"].median()),
            average_expanded_states=float(group["expanded_states"].mean()),
            average_maximum_frontier_size=float(
                group["maximum_frontier_size"].mean()
            ),
            optimal_path_rate=float(group["matches_dataset_optimal_steps"].mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    with np.load(args.dataset, allow_pickle=True) as data:
        starts = np.asarray(data["start_cells"], dtype=np.int16)
        goals = np.asarray(data["goal_cells"], dtype=np.int16)
        maze_indices = np.asarray(data["maze_indices"], dtype=np.int32)
        maze_names = np.asarray(data["maze_name_per_episode"]).astype(str)
        episode_ids = np.asarray(data["episode_ids"])
        fold_ids = np.asarray(data["fold_ids"], dtype=np.int16)
        path_lengths = np.asarray(data["path_lengths"], dtype=np.int16)
        action_deltas = np.asarray(data["action_deltas"], dtype=np.int16)
        mazes = [np.asarray(item, dtype=np.int8) for item in data["mazes"]]

    rows: list[dict[str, object]] = []
    for index in range(len(episode_ids)):
        maze = mazes[int(maze_indices[index])]
        start = tuple(int(value) for value in starts[index])
        goal = tuple(int(value) for value in goals[index])
        before = time.perf_counter_ns()
        path, expanded_states, maximum_frontier_size = bfs_shortest_path(
            maze, start, goal, action_deltas
        )
        elapsed_seconds = (time.perf_counter_ns() - before) / 1_000_000_000
        steps = len(path) - 1
        expected_optimal_steps = int(path_lengths[index]) - 1
        rows.append(
            {
                "cv_fold": int(fold_ids[index]),
                "maze_name": maze_names[index],
                "episode_id": int(episode_ids[index]),
                "query_index": index,
                "strategy": "bfs",
                "goal_reached": True,
                "steps_to_goal": steps,
                "dataset_optimal_steps": expected_optimal_steps,
                "matches_dataset_optimal_steps": steps == expected_optimal_steps,
                "compute_seconds": elapsed_seconds,
                "expanded_states": expanded_states,
                "maximum_frontier_size": maximum_frontier_size,
            }
        )

    results = pd.DataFrame(rows)
    if not results["matches_dataset_optimal_steps"].all():
        failures = results.loc[
            ~results["matches_dataset_optimal_steps"],
            ["episode_id", "steps_to_goal", "dataset_optimal_steps"],
        ]
        raise AssertionError(f"BFS disagrees with stored optimal paths:\n{failures}")

    summary = summarize(results, ["strategy"])
    per_maze = summarize(results, ["maze_name", "strategy"])
    per_fold = summarize(results, ["cv_fold", "strategy"])
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    per_maze.to_csv(args.per_maze_summary_csv, index=False)
    per_fold.to_csv(args.per_fold_summary_csv, index=False)
    print(
        json.dumps(
            {
                "queries": len(results),
                "folds": sorted(int(value) for value in results.cv_fold.unique()),
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
