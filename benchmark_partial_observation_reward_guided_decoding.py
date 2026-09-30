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
TYPE_RETURN_TO_GO = 7
STRATEGIES = ("reward_greedy", "reward_beam_2", "reward_beam_3")
METRICS = (
    "invalid_real_action_rate",
    "first_wall_mask_accuracy",
    "first_visible_goal_accuracy",
    "first_goal_row_accuracy",
    "first_goal_column_accuracy",
    "complete_next_observation_accuracy",
    "first_reward_accuracy",
    "first_reward_mae",
    "first_return_to_go_accuracy",
    "first_return_to_go_mae",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cv-fold", type=int, required=True)
    parser.add_argument("--queries-per-maze", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=19)
    parser.add_argument("--planning-horizon", type=int, default=5)
    parser.add_argument("--observation-proposals-per-action", type=int, default=3)
    parser.add_argument("--mazes", nargs="+", required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--per-maze-summary-csv", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def metadata_scalar(metadata: dict, key: str) -> int:
    if key not in metadata:
        raise KeyError(f"Checkpoint is missing dataset metadata {key!r}")
    value = metadata[key]
    if isinstance(value, list):
        value = value[0]
    return int(value)


def trim_complete_groups(
    tokens: list[int], types: list[int], maximum: int
) -> tuple[list[int], list[int]]:
    if len(tokens) <= maximum:
        return tokens, types
    prefix_size = 5
    transition_size = 7
    available = max(0, maximum - prefix_size)
    keep = (available // transition_size) * transition_size
    if keep == 0:
        return tokens[:maximum], types[:maximum]
    return tokens[:prefix_size] + tokens[-keep:], types[:prefix_size] + types[-keep:]


def batched_logits(model, token_batches, type_batches, device):
    maximum = int(model.pos_encoder.pe.shape[1])
    trimmed = [
        trim_complete_groups(list(tokens), list(types), maximum)
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
        logits = model(input_ids, type_ids, attention)
    return torch.stack([logits[row, position] for row, position in enumerate(final_positions)])


def top_observation_proposals(
    model,
    action_candidates,
    wall_offset,
    goal_offset,
    delta_offset,
    delta_min,
    delta_count,
    observation_width,
    device,
):
    action_logits = batched_logits(
        model,
        [candidate["tokens"] for candidate in action_candidates],
        [candidate["types"] for candidate in action_candidates],
        device,
    )
    wall_candidates = []
    for candidate_id, (candidate, logits) in enumerate(zip(action_candidates, action_logits)):
        scores = torch.log_softmax(logits[wall_offset:wall_offset + 512], dim=-1)
        count = min(observation_width, 512)
        values, positions = torch.topk(scores, count)
        for score, wall in zip(values.tolist(), positions.tolist()):
            wall_candidates.append({
                **candidate,
                "action_candidate_id": candidate_id,
                "tokens": candidate["tokens"] + [wall_offset + int(wall)],
                "types": candidate["types"] + [TYPE_WALLS],
                "likelihood": candidate["likelihood"] + float(score),
                "predicted_wall": int(wall),
            })

    visible_logits = batched_logits(
        model,
        [candidate["tokens"] for candidate in wall_candidates],
        [candidate["types"] for candidate in wall_candidates],
        device,
    )
    grouped: dict[int, list[dict]] = {}
    for candidate, logits in zip(wall_candidates, visible_logits):
        scores = torch.log_softmax(logits[goal_offset:goal_offset + 10], dim=-1)
        for visible, score in enumerate(scores.tolist()):
            extension = {
                **candidate,
                "tokens": candidate["tokens"] + [goal_offset + visible],
                "types": candidate["types"] + [TYPE_VISIBLE_GOAL],
                "likelihood": candidate["likelihood"] + float(score),
                "predicted_visible": int(visible),
            }
            grouped.setdefault(candidate["action_candidate_id"], []).append(extension)
    visible_candidates = []
    for candidates in grouped.values():
        candidates.sort(key=lambda item: (-item["likelihood"], item["tokens"]))
        visible_candidates.extend(candidates[:observation_width])

    row_logits = batched_logits(
        model,
        [candidate["tokens"] for candidate in visible_candidates],
        [candidate["types"] for candidate in visible_candidates],
        device,
    )
    row_candidates = []
    upper = delta_offset + delta_count
    for candidate, logits in zip(visible_candidates, row_logits):
        action = candidate["selected_action"]
        dr, dc = candidate["action_delta"]
        next_row = candidate["row"] - dr
        next_col = candidate["col"] - dc
        row_token = delta_offset + next_row - delta_min
        if not delta_offset <= row_token < upper:
            continue
        scores = torch.log_softmax(logits[delta_offset:upper], dim=-1)
        row_candidates.append({
            **candidate,
            "tokens": candidate["tokens"] + [row_token],
            "types": candidate["types"] + [TYPE_GOAL_DELTA],
            "likelihood": candidate["likelihood"] + float(scores[row_token - delta_offset]),
            "next_row": next_row,
            "next_col": next_col,
            "row_token": row_token,
        })
    if not row_candidates:
        return []

    column_logits = batched_logits(
        model,
        [candidate["tokens"] for candidate in row_candidates],
        [candidate["types"] for candidate in row_candidates],
        device,
    )
    complete = []
    for candidate, logits in zip(row_candidates, column_logits):
        column_token = delta_offset + candidate["next_col"] - delta_min
        if not delta_offset <= column_token < upper:
            continue
        scores = torch.log_softmax(logits[delta_offset:upper], dim=-1)
        observation = [
            wall_offset + candidate["predicted_wall"],
            goal_offset + candidate["predicted_visible"],
            candidate["row_token"],
            column_token,
        ]
        complete.append({
            **candidate,
            "tokens": candidate["tokens"] + [column_token],
            "types": candidate["types"] + [TYPE_GOAL_DELTA],
            "likelihood": candidate["likelihood"] + float(scores[column_token - delta_offset]),
            "next_observation": observation,
        })
    return complete


def add_reward_predictions(
    model,
    candidates,
    reward_offset,
    reward_values,
    return_offset,
    return_values,
    device,
):
    reward_logits = batched_logits(
        model,
        [candidate["tokens"] for candidate in candidates],
        [candidate["types"] for candidate in candidates],
        device,
    )
    reward_prefixes = []
    reward_values_tensor = torch.as_tensor(reward_values, dtype=torch.float32, device=device)
    for candidate, logits in zip(candidates, reward_logits):
        distribution = torch.softmax(
            logits[reward_offset:reward_offset + len(reward_values)], dim=-1
        )
        expected_reward = float((distribution * reward_values_tensor).sum().item())
        reward_index = int(distribution.argmax().item())
        reward_prefixes.append({
            **candidate,
            "tokens": candidate["tokens"] + [reward_offset + reward_index],
            "types": candidate["types"] + [TYPE_REWARD],
            "likelihood": candidate["likelihood"] + float(torch.log(distribution[reward_index]).item()),
            "predicted_reward": expected_reward,
            "predicted_reward_index": reward_index,
        })

    return_logits = batched_logits(
        model,
        [candidate["tokens"] for candidate in reward_prefixes],
        [candidate["types"] for candidate in reward_prefixes],
        device,
    )
    completed = []
    return_values_tensor = torch.as_tensor(return_values, dtype=torch.float32, device=device)
    for candidate, logits in zip(reward_prefixes, return_logits):
        distribution = torch.softmax(
            logits[return_offset:return_offset + len(return_values)], dim=-1
        )
        expected_return = float((distribution * return_values_tensor).sum().item())
        return_index = int(distribution.argmax().item())
        completed.append({
            **candidate,
            "tokens": candidate["tokens"] + [return_offset + return_index],
            "types": candidate["types"] + [TYPE_RETURN_TO_GO],
            "likelihood": candidate["likelihood"] + float(torch.log(distribution[return_index]).item()),
            "predicted_return": expected_return,
            "predicted_return_index": return_index,
            "return_score": candidate["cumulative_reward"] + expected_return,
        })
    return completed


def plan_reward_guided(
    model,
    history,
    history_types,
    action_deltas,
    observation_offsets,
    reward_metadata,
    horizon,
    width,
    observation_width,
    device,
):
    action_offset, wall_offset, goal_offset, delta_offset, delta_min, delta_count = observation_offsets
    reward_offset, reward_values, return_offset, return_values = reward_metadata
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
        "likelihood": 0.0,
        "cumulative_reward": 0.0,
        "return_score": float("-inf"),
        "actions": [],
        "observations": [],
        "done": False,
        "first_transition": None,
    }]

    for _ in range(horizon):
        finished = [beam for beam in beams if beam["done"]]
        active = [beam for beam in beams if not beam["done"]]
        if not active:
            break

        logits_batch = batched_logits(
            model,
            [beam["tokens"] for beam in active],
            [beam["types"] for beam in active],
            device,
        )
        action_candidates = []
        for beam, logits in zip(active, logits_batch):
            scores = torch.log_softmax(
                logits[action_offset:action_offset + len(action_deltas)], dim=-1
            )
            for action in observed_action_indices(beam["wall"], action_deltas):
                dr, dc = (int(value) for value in action_deltas[action])
                action_candidates.append({
                    **beam,
                    "tokens": beam["tokens"] + [action_offset + action],
                    "types": beam["types"] + [TYPE_ACTION],
                    "likelihood": beam["likelihood"] + float(scores[action]),
                    "selected_action": action,
                    "action_delta": (dr, dc),
                    "actions": beam["actions"] + [action],
                })
        if not action_candidates:
            break

        proposals = top_observation_proposals(
            model,
            action_candidates,
            wall_offset,
            goal_offset,
            delta_offset,
            delta_min,
            delta_count,
            observation_width,
            device,
        )
        if not proposals:
            break
        proposals = add_reward_predictions(
            model,
            proposals,
            reward_offset,
            reward_values,
            return_offset,
            return_values,
            device,
        )
        completed = []
        for proposal in proposals:
            first_transition = proposal["first_transition"]
            if first_transition is None:
                first_transition = {
                    "action": proposal["selected_action"],
                    "observation": proposal["next_observation"],
                    "reward_index": proposal["predicted_reward_index"],
                    "reward": proposal["predicted_reward"],
                    "return_index": proposal["predicted_return_index"],
                    "return": proposal["predicted_return"],
                }
            completed.append({
                **proposal,
                "wall": proposal["predicted_wall"],
                "row": proposal["next_row"],
                "col": proposal["next_col"],
                "cumulative_reward": (
                    proposal["cumulative_reward"] + proposal["predicted_reward"]
                ),
                "observations": proposal["observations"] + [proposal["next_observation"]],
                "done": proposal["next_row"] == 0 and proposal["next_col"] == 0,
                "first_transition": first_transition,
            })
        beams = sorted(
            finished + completed,
            key=lambda beam: (-beam["return_score"], -beam["likelihood"], beam["tokens"]),
        )[:width]

    if not beams or beams[0]["first_transition"] is None:
        raise RuntimeError("No reward-guided action could be planned")
    chosen = sorted(
        beams,
        key=lambda beam: (-beam["return_score"], -beam["likelihood"], beam["tokens"]),
    )[0]
    return chosen["first_transition"]


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
    return_offset = metadata_scalar(metadata, "return_to_go_token_offset")
    return_min = metadata_scalar(metadata, "return_to_go_min_cents")
    return_max = metadata_scalar(metadata, "return_to_go_max_cents")
    return_values = np.arange(return_min, return_max + 1, dtype=np.float32) / reward_scale
    if metadata_scalar(metadata, "token_type_reward") != TYPE_REWARD:
        raise ValueError("Unexpected reward token type in checkpoint")
    if metadata_scalar(metadata, "token_type_return_to_go") != TYPE_RETURN_TO_GO:
        raise ValueError("Unexpected return-to-go token type in checkpoint")
    return reward_offset, reward_values, return_offset, return_values


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
    observation_width,
    device,
):
    observation_offsets = offsets_from(data)
    action_offset, wall_offset, _, _, _, _ = observation_offsets
    reward_offset, reward_values, return_offset, return_values = reward_metadata
    bos = int(data["bos_token_id"][0])
    current = start.copy()
    initial_observation = observation_tokens(data, maze, current, goal)
    history = [bos] + initial_observation
    history_types = [1, TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA]
    component_correct = [[], [], [], []]
    complete_correct = []
    reward_correct = []
    reward_errors = []
    predicted_returns = []
    predicted_return_indices = []
    realised_rewards = []
    invalid = 0
    path = [current.copy()]

    for step in range(max_steps):
        predicted = plan_reward_guided(
            model,
            history,
            history_types,
            action_deltas,
            observation_offsets,
            reward_metadata,
            horizon,
            width,
            observation_width,
            device,
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

        real_observation = observation_tokens(data, maze, current, goal)
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
        predicted_returns.append(float(predicted["return"]))
        predicted_return_indices.append(int(predicted["return_index"]))

        history.extend(
            [action_offset + action]
            + real_observation
            + [reward_offset + actual_reward_index, return_offset + int(predicted["return_index"])]
        )
        history_types.extend([
            TYPE_ACTION,
            TYPE_WALLS,
            TYPE_VISIBLE_GOAL,
            TYPE_GOAL_DELTA,
            TYPE_GOAL_DELTA,
            TYPE_REWARD,
            TYPE_RETURN_TO_GO,
        ])
        path.append(current.copy())
        if reached_goal:
            break

    realised_returns = np.cumsum(np.asarray(realised_rewards)[::-1])[::-1]
    return_errors = [
        abs(predicted - realised)
        for predicted, realised in zip(predicted_returns, realised_returns)
    ]
    actual_return_indices = [
        int(np.argmin(np.abs(return_values - realised))) for realised in realised_returns
    ]
    return_correct = [
        predicted == actual
        for predicted, actual in zip(predicted_return_indices, actual_return_indices)
    ]
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    diagnostics = {
        "first_wall_mask_accuracy": mean(component_correct[0]),
        "first_visible_goal_accuracy": mean(component_correct[1]),
        "first_goal_row_accuracy": mean(component_correct[2]),
        "first_goal_column_accuracy": mean(component_correct[3]),
        "complete_next_observation_accuracy": mean(complete_correct),
        "first_reward_accuracy": mean(reward_correct),
        "first_reward_mae": mean(reward_errors),
        "first_return_to_go_accuracy": mean(return_correct),
        "first_return_to_go_mae": mean(return_errors),
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
    if args.observation_proposals_per_action < 1:
        raise ValueError("--observation-proposals-per-action must be positive")
    device = choose_device(args.device)
    data = np.load(args.dataset, allow_pickle=True)
    saved = torch.load(args.checkpoint, map_location=device)
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
            beam_rows = existing[existing["strategy"].str.startswith("reward_beam_")]
            if len(beam_rows) and not beam_rows["planning_horizon"].eq(args.planning_horizon).all():
                raise ValueError("Existing results use a different planning horizon")
        rows = existing.to_dict("records")
    completed = {
        (int(row["query_index"]), str(row["strategy"])) for row in rows
    }

    print(json.dumps({
        "device": str(device),
        "fold": args.cv_fold,
        "strategies": args.strategies,
        "observation_proposals_per_action": args.observation_proposals_per_action,
        "existing_rows": len(rows),
    }, indent=2))

    specifications = {
        "reward_greedy": (1, 1, 1),
        "reward_beam_2": (2, args.planning_horizon, args.observation_proposals_per_action),
        "reward_beam_3": (3, args.planning_horizon, args.observation_proposals_per_action),
    }
    for maze_name in args.mazes:
        indices = np.flatnonzero((folds == args.cv_fold) & (names == maze_name))
        order = np.lexsort((
            goals[indices, 1],
            goals[indices, 0],
            starts[indices, 1],
            starts[indices, 0],
            lengths[indices],
        ))
        for index in indices[order][:args.queries_per_maze]:
            maze = mazes[int(maze_indices[index])]
            start = starts[index].astype(np.int16)
            goal = goals[index].astype(np.int16)
            for strategy in args.strategies:
                key = (int(index), strategy)
                if key in completed:
                    continue
                width, horizon, observation_width = specifications[strategy]
                before = time.perf_counter()
                path, invalid, diagnostics = rollout(
                    model,
                    data,
                    maze,
                    start,
                    goal,
                    action_deltas,
                    reward_metadata,
                    args.max_steps,
                    horizon,
                    width,
                    observation_width,
                    device,
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
                    "observation_proposals_per_action": observation_width,
                    "action_expansion": "deterministic_enumeration",
                    "return_discount_factor": 1.0,
                    "real_history_return_token": "selected_model_prediction",
                }
                row.update(diagnostics)
                rows.append(row)
                completed.add(key)
        write_outputs(rows, args)
        print(f"Completed {maze_name}: {len(rows)} rows saved", flush=True)

    print(summarize(pd.DataFrame(rows), ["strategy"]).to_string(index=False))


if __name__ == "__main__":
    main()
