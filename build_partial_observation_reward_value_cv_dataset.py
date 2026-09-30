from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_partial_observation_cv_dataset import local_encoding


DEFAULT_SOURCE = Path("maze2d_partial_obs_3x3_joint_obs_action_mixed3_4act_cv5.npz")
DEFAULT_OUTPUT = Path("maze2d_partial_obs_3x3_joint_obs_action_reward_value_mixed3_4act_cv5.npz")

IGNORE_INDEX = -100
STEP_REWARD_CENTS = -1
GOAL_REWARD_CENTS = 100
FAILURE_REWARD_CENTS = -100
REWARD_VALUES_CENTS = np.asarray(
    [FAILURE_REWARD_CENTS, STEP_REWARD_CENTS, GOAL_REWARD_CENTS], dtype=np.int16
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add immediate-reward tokens and scalar return-to-go targets to the "
            "three-rollout mixed partial-observation dataset. Return-to-go values "
            "are labels for a value head and are not inserted into token sequences."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def scalar(arrays: dict[str, np.ndarray], key: str) -> int:
    return int(np.asarray(arrays[key]).reshape(-1)[0])


def rewards_for_episode(step_count: int, reached_goal: bool) -> np.ndarray:
    if step_count < 1:
        raise ValueError("Every trajectory must contain at least one action")
    rewards = np.full(step_count, STEP_REWARD_CENTS, dtype=np.int16)
    rewards[-1] = GOAL_REWARD_CENTS if reached_goal else FAILURE_REWARD_CENTS
    return rewards


def undiscounted_returns(rewards: np.ndarray) -> np.ndarray:
    return np.cumsum(rewards[::-1], dtype=np.int32)[::-1].astype(np.int16)


def main() -> None:
    args = parse_args()
    with np.load(args.source, allow_pickle=True) as source:
        arrays = {key: source[key] for key in source.files}

    paths = [np.asarray(path, dtype=np.int16) for path in arrays["paths_xy"]]
    goals = np.asarray(arrays["goal_cells"], dtype=np.int16)
    maze_indices = np.asarray(arrays["maze_indices"], dtype=np.int32)
    mazes = [np.asarray(maze, dtype=np.int8) for maze in arrays["mazes"]]
    action_deltas = np.asarray(arrays["action_deltas"], dtype=np.int16)
    reached = np.asarray(arrays["goal_reached"], dtype=np.bool_)
    actual_steps = np.asarray(arrays["actual_steps"], dtype=np.int32)

    pad = scalar(arrays, "pad_token_id")
    bos = scalar(arrays, "bos_token_id")
    old_vocab_size = scalar(arrays, "vocab_size")
    wall_offset = scalar(arrays, "wall_token_offset")
    goal_offset = scalar(arrays, "visible_goal_token_offset")
    delta_offset = scalar(arrays, "goal_delta_token_offset")
    delta_min = scalar(arrays, "goal_delta_min")
    delta_max = scalar(arrays, "goal_delta_max")
    action_offset = scalar(arrays, "action_token_offset")

    type_pad = scalar(arrays, "token_type_pad")
    type_bos = scalar(arrays, "token_type_bos")
    type_walls = scalar(arrays, "token_type_walls")
    type_visible_goal = scalar(arrays, "token_type_visible_goal")
    type_goal_delta = scalar(arrays, "token_type_goal_delta")
    type_action = scalar(arrays, "token_type_action")
    type_reward = max(
        type_pad, type_bos, type_walls, type_visible_goal, type_goal_delta, type_action
    ) + 1

    reward_offset = old_vocab_size
    reward_count = len(REWARD_VALUES_CENTS)
    new_vocab_size = reward_offset + reward_count
    reward_to_index = {
        int(value): index for index, value in enumerate(REWARD_VALUES_CENTS.tolist())
    }
    delta_to_action = {
        tuple(int(value) for value in delta): index
        for index, delta in enumerate(action_deltas)
    }

    sequences: list[np.ndarray] = []
    joint_labels: list[np.ndarray] = []
    action_labels: list[np.ndarray] = []
    observation_labels: list[np.ndarray] = []
    reward_labels: list[np.ndarray] = []
    token_types: list[np.ndarray] = []
    episode_value_targets: list[np.ndarray] = []
    episode_rewards: list[np.ndarray] = []
    episode_returns: list[np.ndarray] = []

    for episode, path in enumerate(paths):
        step_count = len(path) - 1
        if step_count != int(actual_steps[episode]):
            raise AssertionError(f"Episode {episode} has inconsistent step metadata")
        goal = goals[episode]
        maze = mazes[int(maze_indices[episode])]
        rewards = rewards_for_episode(step_count, bool(reached[episode]))
        returns = undiscounted_returns(rewards)

        tokens = [bos]
        types = [type_bos]
        labels = [IGNORE_INDEX]
        labels_action = [IGNORE_INDEX]
        labels_observation = [IGNORE_INDEX]
        labels_reward = [IGNORE_INDEX]
        values = [np.nan]

        wall_mask, visible_goal, _ = local_encoding(maze, path[0], goal)
        delta = goal - path[0]
        initial_observation = [
            wall_offset + wall_mask,
            goal_offset + visible_goal,
            delta_offset + int(delta[0] - delta_min),
            delta_offset + int(delta[1] - delta_min),
        ]
        tokens.extend(initial_observation)
        types.extend([type_walls, type_visible_goal, type_goal_delta, type_goal_delta])
        for target in (labels, labels_action, labels_observation, labels_reward):
            target.extend([IGNORE_INDEX] * 4)
        values.extend([np.nan] * 4)

        for step in range(step_count):
            cell = path[step]
            next_cell = path[step + 1]
            action_index = delta_to_action[tuple(int(value) for value in next_cell - cell)]
            action_token = action_offset + action_index

            tokens.append(action_token)
            types.append(type_action)
            labels.append(action_token)
            labels_action.append(action_token)
            labels_observation.append(IGNORE_INDEX)
            labels_reward.append(IGNORE_INDEX)
            values.append(float(returns[step]) / 100.0)

            wall_mask, visible_goal, _ = local_encoding(maze, next_cell, goal)
            delta = goal - next_cell
            if np.any(delta < delta_min) or np.any(delta > delta_max):
                raise ValueError(f"Episode {episode} has out-of-range goal displacement {delta}")
            observation = [
                wall_offset + wall_mask,
                goal_offset + visible_goal,
                delta_offset + int(delta[0] - delta_min),
                delta_offset + int(delta[1] - delta_min),
            ]
            tokens.extend(observation)
            types.extend([type_walls, type_visible_goal, type_goal_delta, type_goal_delta])
            labels.extend(observation)
            labels_action.extend([IGNORE_INDEX] * 4)
            labels_observation.extend(observation)
            labels_reward.extend([IGNORE_INDEX] * 4)
            values.extend([np.nan] * 4)

            reward_value = int(rewards[step])
            reward_token = reward_offset + reward_to_index[reward_value]
            tokens.append(reward_token)
            types.append(type_reward)
            labels.append(reward_token)
            labels_action.append(IGNORE_INDEX)
            labels_observation.append(IGNORE_INDEX)
            labels_reward.append(reward_token)
            values.append(np.nan)

        sequences.append(np.asarray(tokens, dtype=np.int32))
        joint_labels.append(np.asarray(labels, dtype=np.int32))
        action_labels.append(np.asarray(labels_action, dtype=np.int32))
        observation_labels.append(np.asarray(labels_observation, dtype=np.int32))
        reward_labels.append(np.asarray(labels_reward, dtype=np.int32))
        token_types.append(np.asarray(types, dtype=np.int8))
        episode_value_targets.append(np.asarray(values, dtype=np.float32))
        episode_rewards.append(rewards)
        episode_returns.append(returns)

    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int32)
    episode_count = len(sequences)
    max_length = int(lengths.max())
    input_ids = np.full((episode_count, max_length), pad, dtype=np.int32)
    labels = np.full((episode_count, max_length), IGNORE_INDEX, dtype=np.int32)
    action_targets = np.full_like(labels, IGNORE_INDEX)
    observation_targets = np.full_like(labels, IGNORE_INDEX)
    reward_targets = np.full_like(labels, IGNORE_INDEX)
    value_targets = np.full((episode_count, max_length), np.nan, dtype=np.float32)
    attention_mask = np.zeros((episode_count, max_length), dtype=np.int8)
    type_ids = np.full((episode_count, max_length), type_pad, dtype=np.int8)

    for episode, length in enumerate(lengths):
        stop = int(length)
        input_ids[episode, :stop] = sequences[episode]
        labels[episode, :stop] = joint_labels[episode]
        action_targets[episode, :stop] = action_labels[episode]
        observation_targets[episode, :stop] = observation_labels[episode]
        reward_targets[episode, :stop] = reward_labels[episode]
        value_targets[episode, :stop] = episode_value_targets[episode]
        attention_mask[episode, :stop] = 1
        type_ids[episode, :stop] = token_types[episode]

    replaced_keys = {
        "input_ids", "labels", "joint_labels", "action_labels", "observation_labels",
        "immediate_reward_labels", "return_to_go_labels", "attention_mask",
        "token_type_ids", "sequence_lengths", "supervised_action_mask",
        "supervised_observation_mask", "supervised_reward_mask",
        "supervised_return_to_go_mask", "supervised_value_mask", "value_targets",
        "vocab_size", "reward_token_offset", "reward_token_count", "reward_values_cents",
        "return_to_go_token_offset", "return_to_go_token_count", "return_to_go_min_cents",
        "return_to_go_max_cents", "token_type_reward", "token_type_return_to_go",
        "immediate_rewards_cents", "returns_to_go_cents", "reward_scale",
        "discount_factor", "supervision", "dataset_version", "value_target_name",
        "value_target_units",
    }
    output = {key: value for key, value in arrays.items() if key not in replaced_keys}
    output.update(
        input_ids=input_ids,
        labels=labels,
        joint_labels=labels,
        action_labels=action_targets,
        observation_labels=observation_targets,
        immediate_reward_labels=reward_targets,
        attention_mask=attention_mask,
        token_type_ids=type_ids,
        sequence_lengths=lengths,
        supervised_action_mask=action_targets != IGNORE_INDEX,
        supervised_observation_mask=observation_targets != IGNORE_INDEX,
        supervised_reward_mask=reward_targets != IGNORE_INDEX,
        supervised_value_mask=np.isfinite(value_targets),
        value_targets=value_targets,
        immediate_rewards_cents=np.asarray(episode_rewards, dtype=object),
        returns_to_go_cents=np.asarray(episode_returns, dtype=object),
        vocab_size=np.asarray([new_vocab_size], dtype=np.int32),
        reward_token_offset=np.asarray([reward_offset], dtype=np.int32),
        reward_token_count=np.asarray([reward_count], dtype=np.int16),
        reward_values_cents=REWARD_VALUES_CENTS,
        return_to_go_min_cents=np.asarray([min(int(v.min()) for v in episode_returns)], dtype=np.int16),
        return_to_go_max_cents=np.asarray([max(int(v.max()) for v in episode_returns)], dtype=np.int16),
        token_type_reward=np.asarray([type_reward], dtype=np.int8),
        reward_scale=np.asarray([100], dtype=np.int16),
        discount_factor=np.asarray([1.0], dtype=np.float32),
        value_target_name=np.asarray(["undiscounted_return_to_go"], dtype="<U32"),
        value_target_units=np.asarray(["reward_units"], dtype="<U16"),
        supervision=np.asarray(
            ["actions_post_action_observations_rewards_and_value_head"], dtype="<U64"
        ),
        dataset_version=np.asarray(
            ["partial_obs_3x3_joint_action_observation_reward_value_v1"], dtype="<U72"
        ),
    )

    expected_steps = int(actual_steps.sum())
    checks = {
        "action": int((action_targets != IGNORE_INDEX).sum()),
        "observation": int((observation_targets != IGNORE_INDEX).sum()) // 4,
        "reward": int((reward_targets != IGNORE_INDEX).sum()),
        "value": int(np.isfinite(value_targets).sum()),
    }
    if any(count != expected_steps for count in checks.values()):
        raise AssertionError(f"Target counts do not match source transitions: {checks}")
    if not np.array_equal(attention_mask.sum(axis=1), lengths):
        raise AssertionError("Attention masks do not match sequence lengths")
    for key in (
        "episode_ids", "source_episode_ids", "query_ids", "rollout_indices",
        "fold_ids", "maze_indices", "start_cells", "goal_cells", "actual_steps",
        "goal_reached", "trajectory_quality",
    ):
        if not np.array_equal(output[key], arrays[key]):
            raise AssertionError(f"Source metadata changed: {key}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print("output_dataset:", args.output)
    print("episodes:", episode_count)
    print("maximum_sequence_length:", max_length)
    print("transitions/value_targets:", expected_steps)
    print("reward_values_cents:", REWARD_VALUES_CENTS.tolist())
    print("return_range:", [float(np.nanmin(value_targets)), float(np.nanmax(value_targets))])
    print("discount_factor: 1.0")
    print("vocab_size:", new_vocab_size)


if __name__ == "__main__":
    main()
