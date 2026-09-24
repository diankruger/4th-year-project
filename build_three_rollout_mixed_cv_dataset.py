from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path("maze2d_all_mazes_transformer_ready_4act_cv5.npz")
DEFAULT_OUTPUT = Path("maze2d_all_mazes_transformer_ready_mixed3_4act_cv5.npz")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate independent epsilon-greedy trajectories for every source start-goal query."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectories-per-query", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--max-steps", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def neighbors(
    maze: np.ndarray,
    cell: tuple[int, int],
    action_deltas: np.ndarray,
) -> list[tuple[int, int]]:
    result = []
    for delta in action_deltas:
        nxt = (cell[0] + int(delta[0]), cell[1] + int(delta[1]))
        inside = 0 <= nxt[0] < maze.shape[0] and 0 <= nxt[1] < maze.shape[1]
        if inside and int(maze[nxt]) == 0:
            result.append(nxt)
    return result


def distance_map(
    maze: np.ndarray,
    goal: tuple[int, int],
    action_deltas: np.ndarray,
) -> dict[tuple[int, int], int]:
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for nxt in neighbors(maze, cell, action_deltas):
            if nxt not in distances:
                distances[nxt] = distances[cell] + 1
                queue.append(nxt)
    return distances


def epsilon_greedy_rollout(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    distances: dict[tuple[int, int], int],
    action_deltas: np.ndarray,
    epsilon: float,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    cell = start
    path = [start]
    for _ in range(max_steps):
        if cell == goal:
            break
        legal = [nxt for nxt in neighbors(maze, cell, action_deltas) if nxt in distances]
        if not legal:
            break
        best_distance = min(distances[nxt] for nxt in legal)
        best = [nxt for nxt in legal if distances[nxt] == best_distance]
        choices = legal if rng.random() < epsilon else best
        cell = choices[int(rng.integers(len(choices)))]
        path.append(cell)
    return np.asarray(path, dtype=np.int16), cell == goal


def state_token(
    cell: np.ndarray,
    state_offset: int,
    global_shape: tuple[int, int],
) -> int:
    return state_offset + int(cell[0]) * global_shape[1] + int(cell[1])


def tokenize_episode(
    maze_token: int,
    path: np.ndarray,
    goal: np.ndarray,
    action_deltas: np.ndarray,
    action_offset: int,
    state_offset: int,
    global_shape: tuple[int, int],
    type_maze: int,
    type_start: int,
    type_goal: int,
    type_state: int,
    type_action: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = [
        maze_token,
        state_token(path[0], state_offset, global_shape),
        state_token(goal, state_offset, global_shape),
        state_token(path[0], state_offset, global_shape),
    ]
    types = [type_maze, type_start, type_goal, type_state]
    labels = [-100, -100, -100, -100]
    delta_to_action = {
        tuple(int(value) for value in delta): action_offset + index
        for index, delta in enumerate(action_deltas)
    }
    for current, nxt in zip(path[:-1], path[1:]):
        delta = tuple(int(value) for value in (nxt - current))
        action_token = delta_to_action[delta]
        inputs.extend([action_token, state_token(nxt, state_offset, global_shape)])
        types.extend([type_action, type_state])
        labels.extend([action_token, -100])
    return (
        np.asarray(inputs, dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(types, dtype=np.int8),
    )


def scalar(data, key: str) -> int:
    return int(np.asarray(data[key]).reshape(-1)[0])


def build(args: argparse.Namespace) -> None:
    if args.trajectories_per_query < 1:
        raise ValueError("--trajectories-per-query must be positive")
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("--epsilon must lie in [0, 1]")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    with np.load(args.source, allow_pickle=True) as source:
        source_arrays = {key: source[key] for key in source.files}

    mazes = [np.asarray(maze, dtype=np.int8) for maze in source_arrays["mazes"]]
    action_deltas = np.asarray(source_arrays["action_deltas"], dtype=np.int16)
    starts = np.asarray(source_arrays["start_cells"], dtype=np.int16)
    goals = np.asarray(source_arrays["goal_cells"], dtype=np.int16)
    maze_indices = np.asarray(source_arrays["maze_indices"], dtype=np.int32)
    source_episode_ids = np.asarray(source_arrays["episode_ids"], dtype=np.int32)
    source_paths = [np.asarray(path, dtype=np.int16) for path in source_arrays["paths_xy"]]
    maze_tokens = np.asarray(source_arrays["maze_token_ids"], dtype=np.int32)
    global_shape = tuple(int(value) for value in source_arrays["global_grid_shape"])
    action_offset = scalar(source_arrays, "action_token_offset")
    state_offset = scalar(source_arrays, "state_token_offset")
    token_types = {
        name: scalar(source_arrays, f"token_type_{name}")
        for name in ("maze", "start", "goal", "state", "action")
    }

    rows = []
    distance_cache: dict[tuple[int, tuple[int, int]], dict[tuple[int, int], int]] = {}
    for query_index, (start_array, goal_array, maze_index) in enumerate(
        zip(starts, goals, maze_indices)
    ):
        start = tuple(int(value) for value in start_array)
        goal = tuple(int(value) for value in goal_array)
        maze_index = int(maze_index)
        maze = mazes[maze_index]
        cache_key = (maze_index, goal)
        if cache_key not in distance_cache:
            distance_cache[cache_key] = distance_map(maze, goal, action_deltas)
        distances = distance_cache[cache_key]
        optimal_steps = len(source_paths[query_index]) - 1

        for rollout_index in range(args.trajectories_per_query):
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, query_index, rollout_index])
            )
            path, reached = epsilon_greedy_rollout(
                maze,
                start,
                goal,
                distances,
                action_deltas,
                args.epsilon,
                args.max_steps,
                rng,
            )
            actual_steps = len(path) - 1
            if not reached:
                quality = "failed"
            elif actual_steps == optimal_steps:
                quality = "optimal"
            else:
                quality = "suboptimal"
            input_ids, labels, types = tokenize_episode(
                int(maze_tokens[maze_index]),
                path,
                goal_array,
                action_deltas,
                action_offset,
                state_offset,
                global_shape,
                token_types["maze"],
                token_types["start"],
                token_types["goal"],
                token_types["state"],
                token_types["action"],
            )
            rows.append({
                "query_index": query_index,
                "rollout_index": rollout_index,
                "source_episode_id": int(source_episode_ids[query_index]),
                "path": path,
                "input_ids": input_ids,
                "labels": labels,
                "types": types,
                "quality": quality,
                "reached": reached,
                "optimal_steps": optimal_steps,
                "actual_steps": actual_steps,
            })

    sequence_lengths = np.asarray([len(row["input_ids"]) for row in rows], dtype=np.int32)
    row_count = len(rows)
    max_length = int(sequence_lengths.max())
    pad_id = scalar(source_arrays, "pad_token_id")
    pad_type = scalar(source_arrays, "token_type_pad")
    input_ids = np.full((row_count, max_length), pad_id, dtype=np.int32)
    labels = np.full((row_count, max_length), -100, dtype=np.int32)
    attention_mask = np.zeros((row_count, max_length), dtype=np.int8)
    token_type_ids = np.full((row_count, max_length), pad_type, dtype=np.int8)
    for row_index, row in enumerate(rows):
        length = int(sequence_lengths[row_index])
        input_ids[row_index, :length] = row["input_ids"]
        labels[row_index, :length] = row["labels"]
        token_type_ids[row_index, :length] = row["types"]
        attention_mask[row_index, :length] = 1

    source_indices = np.asarray([row["query_index"] for row in rows], dtype=np.int32)
    rollout_indices = np.asarray([row["rollout_index"] for row in rows], dtype=np.int16)
    repeated = lambda key: np.asarray(source_arrays[key])[source_indices]
    reached = np.asarray([row["reached"] for row in rows], dtype=np.bool_)
    optimal_steps = np.asarray([row["optimal_steps"] for row in rows], dtype=np.int32)
    actual_steps = np.asarray([row["actual_steps"] for row in rows], dtype=np.int32)
    path_stretch = np.full(row_count, np.nan, dtype=np.float32)
    path_stretch[reached] = actual_steps[reached] / optimal_steps[reached]

    per_episode_keys = {
        "maze_indices",
        "maze_name_per_episode",
        "grid_shape_per_episode",
        "start_cells",
        "goal_cells",
        "fold_ids",
    }
    output = {
        key: repeated(key) if key in per_episode_keys else value
        for key, value in source_arrays.items()
        if key not in {
            "input_ids", "labels", "attention_mask", "token_type_ids", "sequence_lengths",
            "path_lengths", "episode_ids", "paths_xy",
        }
    }
    output.update({
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "sequence_lengths": sequence_lengths,
        "path_lengths": actual_steps + 1,
        "episode_ids": np.arange(row_count, dtype=np.int32),
        "source_episode_ids": np.asarray([row["source_episode_id"] for row in rows], dtype=np.int32),
        "query_ids": source_indices,
        "rollout_indices": rollout_indices,
        "paths_xy": np.asarray([row["path"] for row in rows], dtype=object),
        "trajectory_quality": np.asarray([row["quality"] for row in rows], dtype="<U16"),
        "generator_type": np.full(row_count, "epsilon_greedy", dtype="<U32"),
        "generation_epsilon": np.full(row_count, args.epsilon, dtype=np.float32),
        "goal_reached": reached,
        "optimal_steps": optimal_steps,
        "actual_steps": actual_steps,
        "excess_steps": np.where(reached, actual_steps - optimal_steps, -1).astype(np.int32),
        "path_stretch": path_stretch,
        "mixed_generation_seed": np.asarray([args.seed], dtype=np.int64),
        "mixed_max_steps": np.asarray([args.max_steps], dtype=np.int32),
        "trajectories_per_query": np.asarray([args.trajectories_per_query], dtype=np.int16),
    })

    expected_rows = len(starts) * args.trajectories_per_query
    if row_count != expected_rows:
        raise AssertionError((row_count, expected_rows))
    if not np.all(np.bincount(source_indices, minlength=len(starts)) == args.trajectories_per_query):
        raise AssertionError("Every source query must have exactly the requested number of trajectories")
    expected_rollout_indices = np.tile(
        np.arange(args.trajectories_per_query, dtype=np.int16), len(starts)
    )
    if not np.array_equal(rollout_indices, expected_rollout_indices):
        raise AssertionError("Rollout indices are not grouped correctly")
    if not np.array_equal(output["start_cells"], starts[source_indices]):
        raise AssertionError("Start-cell metadata changed")
    if not np.array_equal(output["goal_cells"], goals[source_indices]):
        raise AssertionError("Goal-cell metadata changed")
    if not np.array_equal(output["fold_ids"], source_arrays["fold_ids"][source_indices]):
        raise AssertionError("Fold assignments changed")
    if not np.array_equal(attention_mask.sum(axis=1), sequence_lengths):
        raise AssertionError("Attention masks do not match sequence lengths")
    if not np.array_equal((labels != -100).sum(axis=1), actual_steps):
        raise AssertionError("Action supervision count does not match trajectory length")
    for row_index, path in enumerate(output["paths_xy"]):
        if not np.array_equal(path[0], output["start_cells"][row_index]):
            raise AssertionError(f"Path {row_index} does not begin at its source start cell")
        ended_at_goal = np.array_equal(path[-1], output["goal_cells"][row_index])
        if ended_at_goal != bool(reached[row_index]):
            raise AssertionError(f"Path {row_index} has inconsistent completion metadata")
        maze = mazes[int(output["maze_indices"][row_index])]
        if any(int(maze[tuple(cell)]) != 0 for cell in path):
            raise AssertionError(f"Path {row_index} enters a wall")
        if len(path) > 1:
            steps = np.diff(path, axis=0)
            if any(not np.any(np.all(action_deltas == step, axis=1)) for step in steps):
                raise AssertionError(f"Path {row_index} contains an invalid action")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)

    qualities, counts = np.unique(output["trajectory_quality"], return_counts=True)
    count_by_quality = dict(zip(qualities.tolist(), counts.tolist()))
    print("output_dataset:", args.output)
    print("source_queries:", len(starts))
    print("trajectories_per_query:", args.trajectories_per_query)
    print("records:", row_count)
    print("fold_counts:", np.bincount(output["fold_ids"]).tolist())
    for quality in ("optimal", "suboptimal", "failed"):
        count = int(count_by_quality.get(quality, 0))
        print(f"{quality}: {count} ({count / row_count:.2%})")


def main() -> None:
    build(arguments())


if __name__ == "__main__":
    main()
