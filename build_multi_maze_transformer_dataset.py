from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


SOURCE_MAZE_DATASET = Path("maze2d_discrete_8x8_dataset.npz")
OUTPUT_DATASET = Path("maze2d_all_mazes_transformer_ready.npz")

PAD_TOKEN_ID = 0
ACTION_TOKEN_NAMES = np.array(["UP", "DOWN", "LEFT", "RIGHT", "STAY"])
ACTION_DELTAS = np.asarray(
    [
        [0, 1],
        [0, -1],
        [-1, 0],
        [1, 0],
        [0, 0],
    ],
    dtype=np.int8,
)
TOKEN_TYPE_PAD = 0
TOKEN_TYPE_MAZE = 1
TOKEN_TYPE_START = 2
TOKEN_TYPE_GOAL = 3
TOKEN_TYPE_STATE = 4
TOKEN_TYPE_ACTION = 5


def cell_to_state_index(cell_xy: np.ndarray, global_grid_shape: tuple[int, int]) -> int:
    return int(cell_xy[0]) * int(global_grid_shape[1]) + int(cell_xy[1])


def bfs_shortest_path(maze: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray | None:
    if start == goal:
        return np.asarray([start], dtype=np.int16)

    rows, cols = maze.shape
    queue: deque[tuple[int, int]] = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    neighbor_order = ((0, 1), (0, -1), (-1, 0), (1, 0))

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dx, dy in neighbor_order:
            nxt = (current[0] + dx, current[1] + dy)
            inside = 0 <= nxt[0] < rows and 0 <= nxt[1] < cols
            if not inside or maze[nxt] != 0 or nxt in parents:
                continue
            parents[nxt] = current
            queue.append(nxt)

    if goal not in parents:
        return None

    path_rev = []
    node: tuple[int, int] | None = goal
    while node is not None:
        path_rev.append(node)
        node = parents[node]
    path_rev.reverse()
    return np.asarray(path_rev, dtype=np.int16)


def build_tokenized_episode(
    maze_token_id: int,
    path_cells: np.ndarray,
    goal_cell: np.ndarray,
    global_grid_shape: tuple[int, int],
    state_token_offset: int,
    action_token_offset: int,
) -> dict[str, np.ndarray | int]:
    def state_token(cell_xy: np.ndarray) -> int:
        return state_token_offset + cell_to_state_index(cell_xy, global_grid_shape)

    obs_xy = np.asarray(path_cells, dtype=np.int16)
    actions = np.diff(obs_xy, axis=0).astype(np.int8) if len(obs_xy) > 1 else np.zeros((0, 2), dtype=np.int8)
    action_delta_to_token = {
        tuple(int(v) for v in delta): int(action_token_offset + idx)
        for idx, delta in enumerate(ACTION_DELTAS)
    }

    input_ids = [maze_token_id, state_token(obs_xy[0]), state_token(goal_cell), state_token(obs_xy[0])]
    token_type_ids = [TOKEN_TYPE_MAZE, TOKEN_TYPE_START, TOKEN_TYPE_GOAL, TOKEN_TYPE_STATE]
    labels = [-100, -100, -100, -100]

    for step in range(len(obs_xy) - 1):
        action_key = tuple(int(v) for v in actions[step])
        action_token = action_delta_to_token[action_key]
        next_state_token = state_token(obs_xy[step + 1])
        input_ids.extend([action_token, next_state_token])
        token_type_ids.extend([TOKEN_TYPE_ACTION, TOKEN_TYPE_STATE])
        labels.extend([action_token, -100])

    return {
        "input_ids": np.asarray(input_ids, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int32),
        "token_type_ids": np.asarray(token_type_ids, dtype=np.int8),
        "sequence_length": int(len(input_ids)),
        "path_length": int(len(obs_xy)),
    }


def main() -> None:
    dataset = np.load(SOURCE_MAZE_DATASET, allow_pickle=True)
    maze_names = np.asarray(dataset["maze_names"])
    mazes = dataset["mazes"]
    grid_shapes = np.asarray(dataset["grid_shapes"], dtype=np.int32)

    num_mazes = int(len(maze_names))
    global_grid_shape = (
        int(grid_shapes[:, 0].max()),
        int(grid_shapes[:, 1].max()),
    )

    maze_token_offset = 1
    state_token_offset = maze_token_offset + num_mazes
    num_state_tokens = int(global_grid_shape[0] * global_grid_shape[1])
    action_token_offset = state_token_offset + num_state_tokens
    vocab_size = action_token_offset + len(ACTION_DELTAS)

    episode_input_ids: list[np.ndarray] = []
    episode_labels: list[np.ndarray] = []
    episode_token_types: list[np.ndarray] = []
    sequence_lengths: list[int] = []
    path_lengths: list[int] = []
    maze_indices: list[int] = []
    maze_name_per_episode: list[str] = []
    grid_shape_per_episode: list[np.ndarray] = []
    start_cells: list[np.ndarray] = []
    goal_cells: list[np.ndarray] = []
    paths_xy: list[np.ndarray] = []

    episode_id = 0
    connected_pair_count = 0
    skipped_pair_count = 0

    for maze_index, (maze_name, maze_obj, grid_shape) in enumerate(zip(maze_names, mazes, grid_shapes)):
        maze = np.asarray(maze_obj, dtype=np.int8)
        free_cells = np.argwhere(maze == 0).astype(np.int16)
        maze_token_id = int(maze_token_offset + maze_index)

        for start_cell in free_cells:
            for goal_cell in free_cells:
                if np.array_equal(start_cell, goal_cell):
                    continue

                path_cells = bfs_shortest_path(
                    maze=maze,
                    start=tuple(int(v) for v in start_cell),
                    goal=tuple(int(v) for v in goal_cell),
                )
                if path_cells is None:
                    skipped_pair_count += 1
                    continue

                packed = build_tokenized_episode(
                    maze_token_id=maze_token_id,
                    path_cells=path_cells,
                    goal_cell=np.asarray(goal_cell, dtype=np.int16),
                    global_grid_shape=global_grid_shape,
                    state_token_offset=state_token_offset,
                    action_token_offset=action_token_offset,
                )
                episode_input_ids.append(packed["input_ids"])
                episode_labels.append(packed["labels"])
                episode_token_types.append(packed["token_type_ids"])
                sequence_lengths.append(int(packed["sequence_length"]))
                path_lengths.append(int(packed["path_length"]))
                maze_indices.append(int(maze_index))
                maze_name_per_episode.append(str(maze_name))
                grid_shape_per_episode.append(np.asarray(grid_shape, dtype=np.int32))
                start_cells.append(np.asarray(start_cell, dtype=np.int16))
                goal_cells.append(np.asarray(goal_cell, dtype=np.int16))
                paths_xy.append(np.asarray(path_cells, dtype=np.int16))
                connected_pair_count += 1
                episode_id += 1

    sequence_lengths_arr = np.asarray(sequence_lengths, dtype=np.int32)
    max_seq_len = int(sequence_lengths_arr.max())
    num_episodes = int(len(episode_input_ids))

    input_ids = np.full((num_episodes, max_seq_len), PAD_TOKEN_ID, dtype=np.int32)
    labels = np.full((num_episodes, max_seq_len), -100, dtype=np.int32)
    attention_mask = np.zeros((num_episodes, max_seq_len), dtype=np.int8)
    token_type_ids = np.full((num_episodes, max_seq_len), TOKEN_TYPE_PAD, dtype=np.int8)

    for idx in range(num_episodes):
        seq_len = int(sequence_lengths_arr[idx])
        input_ids[idx, :seq_len] = episode_input_ids[idx]
        labels[idx, :seq_len] = episode_labels[idx]
        token_type_ids[idx, :seq_len] = episode_token_types[idx]
        attention_mask[idx, :seq_len] = 1

    np.savez_compressed(
        OUTPUT_DATASET,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        sequence_lengths=sequence_lengths_arr,
        path_lengths=np.asarray(path_lengths, dtype=np.int32),
        episode_ids=np.arange(num_episodes, dtype=np.int32),
        maze_indices=np.asarray(maze_indices, dtype=np.int32),
        maze_names=maze_names.astype("<U32"),
        maze_name_per_episode=np.asarray(maze_name_per_episode, dtype="<U32"),
        maze_token_ids=np.arange(maze_token_offset, maze_token_offset + num_mazes, dtype=np.int32),
        mazes=mazes,
        grid_shapes=grid_shapes.astype(np.int32),
        grid_shape_per_episode=np.asarray(grid_shape_per_episode, dtype=np.int32),
        start_cells=np.asarray(start_cells, dtype=np.int16),
        goal_cells=np.asarray(goal_cells, dtype=np.int16),
        paths_xy=np.asarray(paths_xy, dtype=object),
        pad_token_id=np.asarray([PAD_TOKEN_ID], dtype=np.int32),
        vocab_size=np.asarray([vocab_size], dtype=np.int32),
        maze_token_offset=np.asarray([maze_token_offset], dtype=np.int32),
        state_token_offset=np.asarray([state_token_offset], dtype=np.int32),
        action_token_offset=np.asarray([action_token_offset], dtype=np.int32),
        global_grid_shape=np.asarray(global_grid_shape, dtype=np.int32),
        action_token_names=ACTION_TOKEN_NAMES,
        action_deltas=ACTION_DELTAS,
        token_type_pad=np.asarray([TOKEN_TYPE_PAD], dtype=np.int8),
        token_type_maze=np.asarray([TOKEN_TYPE_MAZE], dtype=np.int8),
        token_type_start=np.asarray([TOKEN_TYPE_START], dtype=np.int8),
        token_type_goal=np.asarray([TOKEN_TYPE_GOAL], dtype=np.int8),
        token_type_state=np.asarray([TOKEN_TYPE_STATE], dtype=np.int8),
        token_type_action=np.asarray([TOKEN_TYPE_ACTION], dtype=np.int8),
    )

    print("source_dataset:", SOURCE_MAZE_DATASET)
    print("output_dataset:", OUTPUT_DATASET)
    print("num_mazes:", num_mazes)
    print("maze_names:", [str(name) for name in maze_names])
    print("global_grid_shape:", global_grid_shape)
    print("num_episodes:", num_episodes)
    print("connected_pair_count:", connected_pair_count)
    print("skipped_pair_count:", skipped_pair_count)
    print("max_seq_len:", max_seq_len)
    print("vocab_size:", vocab_size)


if __name__ == "__main__":
    main()
