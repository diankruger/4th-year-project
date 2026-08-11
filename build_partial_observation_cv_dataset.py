from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PAD = 0
BOS = 1
WALL_OFFSET = 2
WALL_COUNT = 512
GOAL_OFFSET = WALL_OFFSET + WALL_COUNT
GOAL_COUNT = 10
DELTA_OFFSET = GOAL_OFFSET + GOAL_COUNT
DELTA_MIN = -11
DELTA_MAX = 11
DELTA_COUNT = DELTA_MAX - DELTA_MIN + 1
ACTION_OFFSET = DELTA_OFFSET + DELTA_COUNT

TYPE_PAD = 0
TYPE_BOS = 1
TYPE_WALLS = 2
TYPE_VISIBLE_GOAL = 3
TYPE_GOAL_DELTA = 4
TYPE_ACTION = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("maze2d_all_mazes_transformer_ready_4act_cv5.npz"))
    parser.add_argument("--output", type=Path, default=Path("maze2d_partial_obs_3x3_optimal_4act_cv5.npz"))
    return parser.parse_args()


def local_encoding(maze: np.ndarray, cell: np.ndarray, goal: np.ndarray) -> tuple[int, int, np.ndarray]:
    patch = np.ones((3, 3), dtype=np.int8)
    visible_goal = 0
    bit = 0
    mask = 0
    for i, dr in enumerate((-1, 0, 1)):
        for j, dc in enumerate((-1, 0, 1)):
            row, col = int(cell[0] + dr), int(cell[1] + dc)
            free = 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1] and int(maze[row, col]) == 0
            patch[i, j] = 0 if free else 1
            if not free:
                mask |= 1 << bit
            if free and row == int(goal[0]) and col == int(goal[1]):
                patch[i, j] = 2
                visible_goal = bit + 1
            bit += 1
    return mask, visible_goal, patch


def main() -> None:
    args = parse_args()
    source = np.load(args.source, allow_pickle=True)
    action_deltas = np.asarray(source["action_deltas"], dtype=np.int8)
    paths = source["paths_xy"]
    goals = np.asarray(source["goal_cells"], dtype=np.int16)
    maze_indices = np.asarray(source["maze_indices"], dtype=np.int32)
    mazes = [np.asarray(value, dtype=np.int8) for value in source["mazes"]]
    sequences, labels, types = [], [], []
    raw_patches, raw_deltas = [], []

    delta_to_action = {tuple(int(x) for x in delta): i for i, delta in enumerate(action_deltas)}
    for episode, path_obj in enumerate(paths):
        path = np.asarray(path_obj, dtype=np.int16)
        goal = goals[episode]
        maze = mazes[int(maze_indices[episode])]
        tokens = [BOS]
        token_types = [TYPE_BOS]
        episode_labels = [-100]
        patches, deltas = [], []
        for step, cell in enumerate(path):
            wall_mask, goal_position, patch = local_encoding(maze, cell, goal)
            delta = goal - cell
            if np.any(delta < DELTA_MIN) or np.any(delta > DELTA_MAX):
                raise ValueError(f"Goal displacement outside [{DELTA_MIN}, {DELTA_MAX}]: {delta}")
            tokens.extend([
                WALL_OFFSET + wall_mask,
                GOAL_OFFSET + goal_position,
                DELTA_OFFSET + int(delta[0] - DELTA_MIN),
                DELTA_OFFSET + int(delta[1] - DELTA_MIN),
            ])
            token_types.extend([TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA])
            episode_labels.extend([-100, -100, -100, -100])
            patches.append(patch)
            deltas.append(delta)
            if step < len(path) - 1:
                action_index = delta_to_action[tuple(int(x) for x in path[step + 1] - cell)]
                action_token = ACTION_OFFSET + action_index
                tokens.append(action_token)
                token_types.append(TYPE_ACTION)
                episode_labels.append(action_token)
        sequences.append(np.asarray(tokens, dtype=np.int32))
        labels.append(np.asarray(episode_labels, dtype=np.int32))
        types.append(np.asarray(token_types, dtype=np.int8))
        raw_patches.append(np.asarray(patches, dtype=np.int8))
        raw_deltas.append(np.asarray(deltas, dtype=np.int16))

    lengths = np.asarray([len(value) for value in sequences], dtype=np.int32)
    max_len = int(lengths.max())
    count = len(sequences)
    input_ids = np.full((count, max_len), PAD, dtype=np.int32)
    action_labels = np.full((count, max_len), -100, dtype=np.int32)
    attention_mask = np.zeros((count, max_len), dtype=np.int8)
    token_type_ids = np.zeros((count, max_len), dtype=np.int8)
    for i, length in enumerate(lengths):
        input_ids[i, :length] = sequences[i]
        action_labels[i, :length] = labels[i]
        attention_mask[i, :length] = 1
        token_type_ids[i, :length] = types[i]

    copied = {key: source[key] for key in source.files if key not in {
        "input_ids", "labels", "attention_mask", "token_type_ids", "sequence_lengths",
        "pad_token_id", "vocab_size", "maze_token_offset", "state_token_offset", "action_token_offset",
        "token_type_pad", "token_type_maze", "token_type_start", "token_type_goal", "token_type_state", "token_type_action"
    }}
    np.savez_compressed(
        args.output,
        **copied,
        input_ids=input_ids,
        labels=action_labels,
        action_labels=action_labels,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        sequence_lengths=lengths,
        local_observations=np.asarray(raw_patches, dtype=object),
        goal_deltas=np.asarray(raw_deltas, dtype=object),
        pad_token_id=np.asarray([PAD], dtype=np.int32),
        vocab_size=np.asarray([ACTION_OFFSET + len(action_deltas)], dtype=np.int32),
        state_token_offset=np.asarray([WALL_OFFSET], dtype=np.int32),
        action_token_offset=np.asarray([ACTION_OFFSET], dtype=np.int32),
        bos_token_id=np.asarray([BOS], dtype=np.int32),
        wall_token_offset=np.asarray([WALL_OFFSET], dtype=np.int32),
        visible_goal_token_offset=np.asarray([GOAL_OFFSET], dtype=np.int32),
        goal_delta_token_offset=np.asarray([DELTA_OFFSET], dtype=np.int32),
        goal_delta_min=np.asarray([DELTA_MIN], dtype=np.int16),
        goal_delta_max=np.asarray([DELTA_MAX], dtype=np.int16),
        observation_encoding=np.asarray(["wall_bitmask_plus_visible_goal_position"], dtype="<U48"),
        token_type_pad=np.asarray([TYPE_PAD], dtype=np.int8),
        token_type_bos=np.asarray([TYPE_BOS], dtype=np.int8),
        token_type_walls=np.asarray([TYPE_WALLS], dtype=np.int8),
        token_type_visible_goal=np.asarray([TYPE_VISIBLE_GOAL], dtype=np.int8),
        token_type_goal_delta=np.asarray([TYPE_GOAL_DELTA], dtype=np.int8),
        token_type_action=np.asarray([TYPE_ACTION], dtype=np.int8),
        dataset_version=np.asarray(["partial_obs_3x3_v1"], dtype="<U32"),
    )
    print(f"wrote {args.output}: episodes={count}, max_seq_len={max_len}, vocab={ACTION_OFFSET + len(action_deltas)}")


if __name__ == "__main__":
    main()
