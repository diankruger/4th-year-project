from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark_partial_observation_joint_receding_cv_decoding import (
    TYPE_ACTION,
    TYPE_GOAL_DELTA,
    TYPE_VISIBLE_GOAL,
    TYPE_WALLS,
    choose_device,
    observation_tokens,
    observed_action_indices,
)
from train_maze2d_discrete_transformer import TinyCausalTransformer


TYPE_REWARD = 6
STRATEGIES = ("value_greedy", "value_beam_2", "value_beam_3")
METRICS = (
    "invalid_real_action_rate",
    "first_wall_mask_accuracy",
    "first_visible_goal_accuracy",
    "first_goal_row_accuracy",
    "first_goal_column_accuracy",
    "complete_next_observation_accuracy",
    "first_reward_accuracy",
    "first_reward_mae",
    "first_value_mae",
    "first_value_bias",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cv-fold", type=int, required=True)
    parser.add_argument("--queries-per-maze", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=19)
    parser.add_argument("--planning-horizon", type=int, default=5)
    parser.add_argument("--likelihood-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--mazes", nargs="+", required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--per-maze-summary-csv", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def metadata_scalar(metadata: dict, key: str) -> int:
    if key not in metadata:
        raise KeyError(f"Checkpoint is missing dataset metadata {key!r}")
    value = metadata[key]
    if isinstance(value, list):
        value = value[0]
    return int(value)


def trim_partial_groups(
    tokens: list[int], types: list[int], maximum: int
) -> tuple[list[int], list[int]]:
    """Keep the fixed prompt, newest complete transitions, and any partial transition."""
    if len(tokens) <= maximum:
        return tokens, types
    prefix_size = 5
    transition_size = 6
    tail_size = len(tokens) - prefix_size
    partial_size = tail_size % transition_size
    available_for_groups = maximum - prefix_size - partial_size
    complete_tokens = max(0, (available_for_groups // transition_size) * transition_size)
    keep = complete_tokens + partial_size
    if keep <= 0:
        return tokens[-maximum:], types[-maximum:]
    return tokens[:prefix_size] + tokens[-keep:], types[:prefix_size] + types[-keep:]


def batched_outputs(model, token_batches, type_batches, device, include_previous=False):
    maximum = int(model.pos_encoder.pe.shape[1])
    trimmed = [
        trim_partial_groups(list(tokens), list(types), maximum)
        for tokens, types in zip(token_batches, type_batches)
    ]
    length = max(len(tokens) for tokens, _ in trimmed)
    input_ids = torch.zeros((len(trimmed), length), dtype=torch.long, device=device)
    type_ids = torch.zeros_like(input_ids)
    attention = torch.zeros_like(input_ids, dtype=torch.bool)
    final_positions = []
    for row, (tokens, types) in enumerate(trimmed):
        count = len(tokens)
        input_ids[row, :count] = torch.as_tensor(tokens, dtype=torch.long, device=device)
        type_ids[row, :count] = torch.as_tensor(types, dtype=torch.long, device=device)
        attention[row, :count] = True
        final_positions.append(count - 1)
    with torch.inference_mode():
        logits, values = model(input_ids, type_ids, attention, return_values=True)
    final_logits = torch.stack([logits[row, position] for row, position in enumerate(final_positions)])
    final_values = torch.stack([values[row, position] for row, position in enumerate(final_positions)])
    if not include_previous:
        return final_logits, final_values
    previous_logits = torch.stack([
        logits[row, position - 1] for row, position in enumerate(final_positions)
    ])
    return final_logits, final_values, previous_logits


def offsets_from(data):
    delta_min = int(data["goal_delta_min"][0])
    delta_max = int(data["goal_delta_max"][0])
    return (
        int(data["action_token_offset"][0]),
        int(data["wall_token_offset"][0]),
        int(data["visible_goal_token_offset"][0]),
        int(data["goal_delta_token_offset"][0]),
        delta_min,
        delta_max - delta_min + 1,
    )


def reward_metadata_from_checkpoint(saved):
    metadata = saved.get("dataset_metadata", {})
    reward_offset = metadata_scalar(metadata, "reward_token_offset")
    reward_scale = metadata_scalar(metadata, "reward_scale")
    reward_values = np.asarray(metadata["reward_values_cents"], dtype=np.float32) / reward_scale
    if metadata_scalar(metadata, "token_type_reward") != TYPE_REWARD:
        raise ValueError("Unexpected reward token type in checkpoint")
    return reward_offset, reward_values


def enumerate_action_candidates(
    model,
    beams,
    action_deltas,
    action_offset,
    likelihood_weight,
    value_weight,
    device,
):
    candidates = []
    for beam in beams:
        for action in observed_action_indices(beam["wall"], action_deltas):
            dr, dc = (int(value) for value in action_deltas[action])
            candidates.append({
                **beam,
                "tokens": beam["tokens"] + [action_offset + action],
                "types": beam["types"] + [TYPE_ACTION],
                "selected_action": int(action),
                "action_delta": (dr, dc),
                "actions": beam["actions"] + [int(action)],
            })
    if not candidates:
        return [], None
    wall_logits, predicted_values, preceding_logits = batched_outputs(
        model,
        [candidate["tokens"] for candidate in candidates],
        [candidate["types"] for candidate in candidates],
        device,
        include_previous=True,
    )
    for row, candidate in enumerate(candidates):
        action_scores = torch.log_softmax(
            preceding_logits[row, action_offset:action_offset + len(action_deltas)], dim=-1
        )
        action_log_probability = float(action_scores[candidate["selected_action"]])
        candidate["model_log_probability"] += action_log_probability
        candidate["predicted_value"] = float(predicted_values[row])
        candidate["value_score"] = (
            candidate["cumulative_reward"] + candidate["predicted_value"]
        )
        candidate["score"] = (
            likelihood_weight * candidate["model_log_probability"]
            + value_weight * candidate["value_score"]
        )
    return candidates, wall_logits


def predict_transition_tokens(
    model,
    candidates,
    wall_logits,
    observation_offsets,
    reward_metadata,
    likelihood_weight,
    value_weight,
    device,
):
    _, wall_offset, goal_offset, delta_offset, delta_min, delta_count = observation_offsets
    reward_offset, reward_values = reward_metadata
    upper = delta_offset + delta_count

    for candidate, logits in zip(candidates, wall_logits):
        scores = torch.log_softmax(logits[wall_offset:wall_offset + 512], dim=-1)
        wall = int(scores.argmax())
        candidate["tokens"] = candidate["tokens"] + [wall_offset + wall]
        candidate["types"] = candidate["types"] + [TYPE_WALLS]
        candidate["model_log_probability"] += float(scores[wall])
        candidate["predicted_wall"] = wall

    visible_logits, _ = batched_outputs(
        model,
        [candidate["tokens"] for candidate in candidates],
        [candidate["types"] for candidate in candidates],
        device,
    )
    for candidate, logits in zip(candidates, visible_logits):
        scores = torch.log_softmax(logits[goal_offset:goal_offset + 10], dim=-1)
        visible = int(scores.argmax())
        candidate["tokens"] = candidate["tokens"] + [goal_offset + visible]
        candidate["types"] = candidate["types"] + [TYPE_VISIBLE_GOAL]
        candidate["model_log_probability"] += float(scores[visible])
        candidate["predicted_visible"] = visible

    row_logits, _ = batched_outputs(
        model,
        [candidate["tokens"] for candidate in candidates],
        [candidate["types"] for candidate in candidates],
        device,
    )
    valid_candidates = []
    for candidate, logits in zip(candidates, row_logits):
        dr, dc = candidate["action_delta"]
        next_row = candidate["row"] - dr
        next_col = candidate["col"] - dc
        row_token = delta_offset + next_row - delta_min
        if not delta_offset <= row_token < upper:
            continue
        scores = torch.log_softmax(logits[delta_offset:upper], dim=-1)
        candidate["tokens"] = candidate["tokens"] + [row_token]
        candidate["types"] = candidate["types"] + [TYPE_GOAL_DELTA]
        candidate["model_log_probability"] += float(scores[row_token - delta_offset])
        candidate["next_row"] = next_row
        candidate["next_col"] = next_col
        candidate["row_token"] = row_token
        valid_candidates.append(candidate)
    candidates = valid_candidates
    if not candidates:
        return []

    column_logits, _ = batched_outputs(
        model,
        [candidate["tokens"] for candidate in candidates],
        [candidate["types"] for candidate in candidates],
        device,
    )
    column_candidates = []
    for candidate, logits in zip(candidates, column_logits):
        column_token = delta_offset + candidate["next_col"] - delta_min
        if not delta_offset <= column_token < upper:
            continue
        scores = torch.log_softmax(logits[delta_offset:upper], dim=-1)
        candidate["tokens"] = candidate["tokens"] + [column_token]
        candidate["types"] = candidate["types"] + [TYPE_GOAL_DELTA]
        candidate["model_log_probability"] += float(scores[column_token - delta_offset])
        candidate["next_observation"] = [
            wall_offset + candidate["predicted_wall"],
            goal_offset + candidate["predicted_visible"],
            candidate["row_token"],
            column_token,
        ]
        column_candidates.append(candidate)
    candidates = column_candidates
    if not candidates:
        return []

    reward_logits, _ = batched_outputs(
        model,
        [candidate["tokens"] for candidate in candidates],
        [candidate["types"] for candidate in candidates],
        device,
    )
    reward_values_tensor = torch.as_tensor(reward_values, dtype=torch.float32, device=device)
    completed = []
    for candidate, logits in zip(candidates, reward_logits):
        distribution = torch.softmax(
            logits[reward_offset:reward_offset + len(reward_values)], dim=-1
        )
        expected_reward = float((distribution * reward_values_tensor).sum())
        reward_index = int(distribution.argmax())
        candidate["tokens"] = candidate["tokens"] + [reward_offset + reward_index]
        candidate["types"] = candidate["types"] + [TYPE_REWARD]
        candidate["model_log_probability"] += float(
            torch.log(distribution[reward_index].clamp_min(1e-12))
        )
        candidate["predicted_reward"] = expected_reward
        candidate["predicted_reward_index"] = reward_index
        candidate["score"] = (
            likelihood_weight * candidate["model_log_probability"]
            + value_weight * candidate["value_score"]
        )
        completed.append(candidate)
    return completed


def plan_value_guided(
    model,
    history,
    history_types,
    action_deltas,
    observation_offsets,
    reward_metadata,
    horizon,
    width,
    likelihood_weight,
    value_weight,
    device,
):
    action_offset, wall_offset, _, delta_offset, delta_min, _ = observation_offsets
    wall_position = max(index for index, token_type in enumerate(history_types) if token_type == TYPE_WALLS)
    delta_positions = [
        index for index, token_type in enumerate(history_types) if token_type == TYPE_GOAL_DELTA
    ][-2:]
    beams = [{
        "tokens": list(history),
        "types": list(history_types),
        "wall": history[wall_position] - wall_offset,
        "row": history[delta_positions[0]] - delta_offset + delta_min,
        "col": history[delta_positions[1]] - delta_offset + delta_min,
        "model_log_probability": 0.0,
        "cumulative_reward": 0.0,
        "value_score": float("-inf"),
        "score": float("-inf"),
        "actions": [],
        "done": False,
        "first_transition": None,
    }]

    for depth in range(horizon):
        finished = [beam for beam in beams if beam["done"]]
        active = [beam for beam in beams if not beam["done"]]
        if not active:
            break
        candidates, wall_logits = enumerate_action_candidates(
            model, active, action_deltas, action_offset,
            likelihood_weight, value_weight, device,
        )
        if not candidates:
            break

        # Greedy requires only the candidate action and its value estimate.
        if horizon == 1 and width == 1:
            chosen = sorted(
                candidates,
                key=lambda item: (-item["score"], -item["model_log_probability"], item["tokens"]),
            )[0]
            return {
                "action": chosen["selected_action"],
                "observation": None,
                "reward_index": None,
                "reward": None,
                "value": chosen["predicted_value"],
            }

        completed = predict_transition_tokens(
            model, candidates, wall_logits, observation_offsets, reward_metadata,
            likelihood_weight, value_weight, device,
        )
        next_beams = []
        for candidate in completed:
            first_transition = candidate["first_transition"]
            if first_transition is None:
                first_transition = {
                    "action": candidate["selected_action"],
                    "observation": candidate["next_observation"],
                    "reward_index": candidate["predicted_reward_index"],
                    "reward": candidate["predicted_reward"],
                    "value": candidate["predicted_value"],
                }
            next_beams.append({
                **candidate,
                "wall": candidate["predicted_wall"],
                "row": candidate["next_row"],
                "col": candidate["next_col"],
                "cumulative_reward": (
                    candidate["cumulative_reward"] + candidate["predicted_reward"]
                ),
                "done": candidate["next_row"] == 0 and candidate["next_col"] == 0,
                "first_transition": first_transition,
            })
        beams = sorted(
            finished + next_beams,
            key=lambda item: (-item["score"], -item["model_log_probability"], item["tokens"]),
        )[:width]

    if not beams or beams[0]["first_transition"] is None:
        raise RuntimeError("No value-guided action could be planned")
    chosen = sorted(
        beams,
        key=lambda item: (-item["score"], -item["model_log_probability"], item["tokens"]),
    )[0]
    return chosen["first_transition"]


def rollout(
    model,
    data,
    maze,
    start,
    goal,
    action_deltas,
    reward_metadata,
    max_steps,
    horizon,
    width,
    likelihood_weight,
    value_weight,
    device,
):
    observation_offsets = offsets_from(data)
    action_offset, _, _, _, _, _ = observation_offsets
    reward_offset, reward_values = reward_metadata
    bos = int(data["bos_token_id"][0])
    current = start.copy()
    initial_observation = observation_tokens(data, maze, current, goal)
    history = [bos] + initial_observation
    history_types = [1, TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA]
    component_correct = [[], [], [], []]
    complete_correct = []
    reward_correct = []
    reward_errors = []
    predicted_values = []
    realised_rewards = []
    invalid = 0
    path = [current.copy()]

    for step in range(max_steps):
        predicted = plan_value_guided(
            model, history, history_types, action_deltas, observation_offsets,
            reward_metadata, horizon, width, likelihood_weight, value_weight, device,
        )
        action = int(predicted["action"])
        proposed = current + action_deltas[action]
        row, column = int(proposed[0]), int(proposed[1])
        valid = (
            0 <= row < maze.shape[0]
            and 0 <= column < maze.shape[1]
            and int(maze[row, column]) == 0
        )
        if valid:
            current = proposed.astype(np.int16)
        else:
            invalid += 1

        reached_goal = bool(np.array_equal(current, goal))
        terminal_failure = step == max_steps - 1 and not reached_goal
        actual_reward = 1.0 if reached_goal else (-1.0 if terminal_failure else -0.01)
        realised_rewards.append(actual_reward)
        predicted_values.append(float(predicted["value"]))
        real_observation = observation_tokens(data, maze, current, goal)

        if predicted["observation"] is not None:
            matches = [
                int(predicted_token) == int(real_token)
                for predicted_token, real_token in zip(predicted["observation"], real_observation)
            ]
            for values, match in zip(component_correct, matches):
                values.append(match)
            complete_correct.append(all(matches))
            actual_reward_index = int(np.argmin(np.abs(reward_values - actual_reward)))
            reward_correct.append(int(predicted["reward_index"]) == actual_reward_index)
            reward_errors.append(abs(float(predicted["reward"]) - actual_reward))
        else:
            actual_reward_index = int(np.argmin(np.abs(reward_values - actual_reward)))

        history.extend(
            [action_offset + action] + real_observation + [reward_offset + actual_reward_index]
        )
        history_types.extend([
            TYPE_ACTION, TYPE_WALLS, TYPE_VISIBLE_GOAL,
            TYPE_GOAL_DELTA, TYPE_GOAL_DELTA, TYPE_REWARD,
        ])
        path.append(current.copy())
        if reached_goal:
            break

    realised_returns = np.cumsum(np.asarray(realised_rewards)[::-1])[::-1]
    value_errors = np.asarray(predicted_values) - realised_returns
    mean = lambda values: float(np.mean(values)) if len(values) else float("nan")
    diagnostics = {
        "first_wall_mask_accuracy": mean(component_correct[0]),
        "first_visible_goal_accuracy": mean(component_correct[1]),
        "first_goal_row_accuracy": mean(component_correct[2]),
        "first_goal_column_accuracy": mean(component_correct[3]),
        "complete_next_observation_accuracy": mean(complete_correct),
        "first_reward_accuracy": mean(reward_correct),
        "first_reward_mae": mean(reward_errors),
        "first_value_mae": mean(np.abs(value_errors)),
        "first_value_bias": mean(value_errors),
    }
    return np.asarray(path), invalid, diagnostics


def summarize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        successful = group.loc[group["goal_reached"].astype(bool), "steps_until_stop"]
        row = dict(zip(columns, keys))
        row.update({
            "completion_rate": float(group["goal_reached"].astype(bool).mean()),
            "average_steps_to_goal": float(successful.mean()) if len(successful) else float("nan"),
            "average_compute_seconds": float(group["compute_seconds"].mean()),
            "average_seconds_per_step": float(group["seconds_per_step"].mean()),
            "invalid_real_action_count": int(group["invalid_real_action_count"].sum()),
        })
        row.update({metric: float(group[metric].mean()) for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(rows: list[dict], args: argparse.Namespace) -> None:
    results = pd.DataFrame(rows).sort_values(
        ["maze_name", "query_index", "strategy"]
    ).reset_index(drop=True)
    summary = summarize(results, ["strategy"]).sort_values("strategy")
    per_maze = summarize(results, ["maze_name", "strategy"]).sort_values(
        ["maze_name", "strategy"]
    )
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    per_maze.to_csv(args.per_maze_summary_csv, index=False)


def main() -> None:
    args = arguments()
    if args.planning_horizon < 1:
        raise ValueError("--planning-horizon must be positive")
    device = choose_device(args.device)
    data = np.load(args.dataset, allow_pickle=True)
    saved = torch.load(args.checkpoint, map_location=device)
    if not bool(saved.get("model_config", {}).get("value_head", False)):
        raise ValueError("Checkpoint does not contain the required value head")
    model = TinyCausalTransformer(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    reward_metadata = reward_metadata_from_checkpoint(saved)

    names = np.asarray(data["maze_name_per_episode"]).astype(str)
    folds = np.asarray(data["fold_ids"])
    starts = np.asarray(data["start_cells"])
    goals = np.asarray(data["goal_cells"])
    lengths = np.asarray(data["path_lengths"])
    maze_indices = np.asarray(data["maze_indices"])
    episode_ids = np.asarray(data["episode_ids"])
    mazes = [np.asarray(value, dtype=np.int8) for value in data["mazes"]]
    action_deltas = np.asarray(data["action_deltas"], dtype=np.int16)

    rows = []
    if args.resume and args.results_csv.is_file():
        existing = pd.read_csv(args.results_csv)
        if len(existing):
            if "value_weight" in existing and not existing["value_weight"].eq(args.value_weight).all():
                raise ValueError("Existing results use a different value weight")
            if "likelihood_weight" in existing and not existing["likelihood_weight"].eq(args.likelihood_weight).all():
                raise ValueError("Existing results use a different likelihood weight")
        rows = existing.to_dict("records")
    completed = {(int(row["query_index"]), str(row["strategy"])) for row in rows}

    query_plan = []
    for maze_name in args.mazes:
        indices = np.flatnonzero((folds == args.cv_fold) & (names == maze_name))
        order = np.lexsort((
            goals[indices, 1], goals[indices, 0], starts[indices, 1],
            starts[indices, 0], lengths[indices],
        ))
        query_plan.extend((maze_name, int(index)) for index in indices[order][:args.queries_per_maze])
    remaining_runs = [
        (maze_name, index, strategy)
        for maze_name, index in query_plan
        for strategy in args.strategies
        if (index, strategy) not in completed
    ]
    total_runs = len(remaining_runs)
    print(json.dumps({
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "fold": args.cv_fold,
        "strategies": args.strategies,
        "planning_horizon": args.planning_horizon,
        "likelihood_weight": args.likelihood_weight,
        "value_weight": args.value_weight,
        "existing_rows": len(rows),
        "remaining_strategy_runs": total_runs,
    }, indent=2))

    specifications = {
        "value_greedy": (1, 1),
        "value_beam_2": (2, args.planning_horizon),
        "value_beam_3": (3, args.planning_horizon),
    }
    started = time.perf_counter()
    completed_now = 0
    for maze_name, index, strategy in remaining_runs:
        maze = mazes[int(maze_indices[index])]
        start = starts[index].astype(np.int16)
        goal = goals[index].astype(np.int16)
        width, horizon = specifications[strategy]
        before = time.perf_counter()
        path, invalid, diagnostics = rollout(
            model, data, maze, start, goal, action_deltas, reward_metadata,
            args.max_steps, horizon, width, args.likelihood_weight,
            args.value_weight, device,
        )
        elapsed = time.perf_counter() - before
        steps = len(path) - 1
        row = {
            "cv_fold": args.cv_fold,
            "maze_name": maze_name,
            "episode_id": int(episode_ids[index]),
            "query_index": int(index),
            "strategy": strategy,
            "steps_until_stop": steps,
            "compute_seconds": elapsed,
            "seconds_per_step": elapsed / max(steps, 1),
            "goal_reached": bool(np.array_equal(path[-1], goal)),
            "invalid_real_action_count": invalid,
            "invalid_real_action_rate": invalid / max(steps, 1),
            "planning_horizon": horizon,
            "beam_width": width,
            "action_expansion": "deterministic_enumeration",
            "likelihood_weight": args.likelihood_weight,
            "value_weight": args.value_weight,
            "return_discount_factor": 1.0,
            "return_values_in_history": False,
        }
        row.update(diagnostics)
        rows.append(row)
        completed_now += 1
        if (
            args.progress_every > 0
            and (completed_now % args.progress_every == 0 or completed_now == total_runs)
        ):
            write_outputs(rows, args)
            total_elapsed = time.perf_counter() - started
            rate = completed_now / max(total_elapsed, 1e-9)
            eta = (total_runs - completed_now) / max(rate, 1e-9)
            print(
                f"evaluation: {completed_now}/{total_runs} strategy runs "
                f"({100.0 * completed_now / max(total_runs, 1):5.1f}%) | "
                f"elapsed {total_elapsed / 60:.1f} min | ETA {eta / 60:.1f} min | "
                f"last={maze_name}/{strategy}",
                flush=True,
            )

    write_outputs(rows, args)
    print(summarize(pd.DataFrame(rows), ["strategy"]).to_string(index=False))


if __name__ == "__main__":
    main()
