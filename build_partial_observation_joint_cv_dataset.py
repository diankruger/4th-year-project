from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_partial_observation_cv_dataset import (
    ACTION_OFFSET, BOS, DELTA_MAX, DELTA_MIN, DELTA_OFFSET, GOAL_OFFSET, PAD,
    TYPE_ACTION, TYPE_BOS, TYPE_GOAL_DELTA, TYPE_PAD, TYPE_VISIBLE_GOAL,
    TYPE_WALLS, WALL_OFFSET, local_encoding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("maze2d_all_mazes_transformer_ready_4act_cv5.npz"))
    parser.add_argument("--output", type=Path, default=Path("maze2d_partial_obs_3x3_joint_obs_action_optimal_4act_cv5.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.source, allow_pickle=True) as source:
        action_deltas = np.asarray(source["action_deltas"], dtype=np.int8)
        paths = source["paths_xy"]
        goals = np.asarray(source["goal_cells"], dtype=np.int16)
        maze_indices = np.asarray(source["maze_indices"], dtype=np.int32)
        mazes = [np.asarray(value, dtype=np.int8) for value in source["mazes"]]
        copied = {key: source[key] for key in source.files if key not in {
            "input_ids", "labels", "attention_mask", "token_type_ids", "sequence_lengths",
            "pad_token_id", "vocab_size", "maze_token_offset", "state_token_offset",
            "action_token_offset", "token_type_pad", "token_type_maze", "token_type_start",
            "token_type_goal", "token_type_state", "token_type_action",
        }}

    delta_to_action = {tuple(int(x) for x in delta): i for i, delta in enumerate(action_deltas)}
    sequences, labels, action_labels, types = [], [], [], []
    raw_patches, raw_deltas = [], []
    for episode, path_obj in enumerate(paths):
        path = np.asarray(path_obj, dtype=np.int16)
        goal = goals[episode]
        maze = mazes[int(maze_indices[episode])]
        tokens, token_types = [BOS], [TYPE_BOS]
        joint, actions_only = [-100], [-100]
        patches, deltas = [], []

        wall_mask, visible_goal, patch = local_encoding(maze, path[0], goal)
        delta = goal - path[0]
        initial = [WALL_OFFSET + wall_mask, GOAL_OFFSET + visible_goal,
                   DELTA_OFFSET + int(delta[0] - DELTA_MIN), DELTA_OFFSET + int(delta[1] - DELTA_MIN)]
        tokens.extend(initial)
        token_types.extend([TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA])
        joint.extend([-100] * 4)
        actions_only.extend([-100] * 4)
        patches.append(patch)
        deltas.append(delta)

        for step in range(len(path) - 1):
            cell, nxt = path[step], path[step + 1]
            action_index = delta_to_action[tuple(int(x) for x in nxt - cell)]
            action_token = ACTION_OFFSET + action_index
            tokens.append(action_token)
            token_types.append(TYPE_ACTION)
            joint.append(action_token)
            actions_only.append(action_token)

            wall_mask, visible_goal, patch = local_encoding(maze, nxt, goal)
            delta = goal - nxt
            if np.any(delta < DELTA_MIN) or np.any(delta > DELTA_MAX):
                raise ValueError(f"Goal displacement outside [{DELTA_MIN}, {DELTA_MAX}]: {delta}")
            observation = [WALL_OFFSET + wall_mask, GOAL_OFFSET + visible_goal,
                           DELTA_OFFSET + int(delta[0] - DELTA_MIN), DELTA_OFFSET + int(delta[1] - DELTA_MIN)]
            tokens.extend(observation)
            token_types.extend([TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA])
            joint.extend(observation)
            actions_only.extend([-100] * 4)
            patches.append(patch)
            deltas.append(delta)

        sequences.append(np.asarray(tokens, dtype=np.int32))
        labels.append(np.asarray(joint, dtype=np.int32))
        action_labels.append(np.asarray(actions_only, dtype=np.int32))
        types.append(np.asarray(token_types, dtype=np.int8))
        raw_patches.append(np.asarray(patches, dtype=np.int8))
        raw_deltas.append(np.asarray(deltas, dtype=np.int16))

    lengths = np.asarray([len(value) for value in sequences], dtype=np.int32)
    count, max_len = len(sequences), int(lengths.max())
    input_ids = np.full((count, max_len), PAD, dtype=np.int32)
    joint_labels = np.full_like(input_ids, -100)
    action_only = np.full_like(input_ids, -100)
    attention = np.zeros((count, max_len), dtype=np.int8)
    token_types = np.full((count, max_len), TYPE_PAD, dtype=np.int8)
    for i, length in enumerate(lengths):
        input_ids[i, :length] = sequences[i]
        joint_labels[i, :length] = labels[i]
        action_only[i, :length] = action_labels[i]
        attention[i, :length] = 1
        token_types[i, :length] = types[i]

    supervised = joint_labels != -100
    np.savez_compressed(
        args.output, **copied, input_ids=input_ids, labels=joint_labels,
        joint_labels=joint_labels, action_labels=action_only, attention_mask=attention,
        token_type_ids=token_types, sequence_lengths=lengths,
        supervised_action_mask=(action_only != -100),
        supervised_observation_mask=supervised & (token_types != TYPE_ACTION),
        local_observations=np.asarray(raw_patches, dtype=object),
        goal_deltas=np.asarray(raw_deltas, dtype=object),
        pad_token_id=np.asarray([PAD], dtype=np.int32),
        vocab_size=np.asarray([ACTION_OFFSET + len(action_deltas)], dtype=np.int32),
        state_token_offset=np.asarray([WALL_OFFSET], dtype=np.int32),
        action_token_offset=np.asarray([ACTION_OFFSET], dtype=np.int32),
        bos_token_id=np.asarray([BOS], dtype=np.int32),
        wall_token_offset=np.asarray([WALL_OFFSET], dtype=np.int32),
        wall_token_count=np.asarray([512], dtype=np.int32),
        visible_goal_token_offset=np.asarray([GOAL_OFFSET], dtype=np.int32),
        visible_goal_token_count=np.asarray([10], dtype=np.int32),
        goal_delta_token_offset=np.asarray([DELTA_OFFSET], dtype=np.int32),
        goal_delta_min=np.asarray([DELTA_MIN], dtype=np.int16),
        goal_delta_max=np.asarray([DELTA_MAX], dtype=np.int16),
        observation_group_size=np.asarray([4], dtype=np.int8),
        observation_encoding=np.asarray(["wall_bitmask_visible_goal_relative_goal"], dtype="<U48"),
        token_type_pad=np.asarray([TYPE_PAD], dtype=np.int8), token_type_bos=np.asarray([TYPE_BOS], dtype=np.int8),
        token_type_walls=np.asarray([TYPE_WALLS], dtype=np.int8),
        token_type_visible_goal=np.asarray([TYPE_VISIBLE_GOAL], dtype=np.int8),
        token_type_goal_delta=np.asarray([TYPE_GOAL_DELTA], dtype=np.int8),
        token_type_action=np.asarray([TYPE_ACTION], dtype=np.int8),
        supervision=np.asarray(["actions_and_post_action_observations"], dtype="<U48"),
        dataset_version=np.asarray(["partial_obs_3x3_joint_action_observation_v1"], dtype="<U56"),
    )
    print(f"wrote {args.output}: episodes={count}, max_seq_len={max_len}, "
          f"action targets={int((action_only != -100).sum())}, "
          f"observation component targets={int((supervised & (token_types != TYPE_ACTION)).sum())}")


if __name__ == "__main__":
    main()
