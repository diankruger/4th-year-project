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
    batched_logits,
    choose_device,
    observation_tokens,
    observed_action_indices,
)
from benchmark_partial_observation_token_level_beam import plan_token_level_beam, retain_best
from train_maze2d_discrete_transformer import TinyCausalTransformer


CONDITIONS = ("predicted_scored", "predicted_context_only", "oracle_context_only")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cv-fold", type=int, required=True)
    parser.add_argument("--queries-per-maze", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=19)
    parser.add_argument("--planning-horizon", type=int, default=5)
    parser.add_argument("--beam-widths", nargs="+", type=int, choices=(2, 3), default=[2, 3])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--mazes", nargs="+", required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--per-maze-summary-csv", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def offsets_from(data) -> tuple[int, int, int, int, int, int]:
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


def plan_predicted_context_only(
    model,
    history,
    history_types,
    action_deltas,
    offsets,
    horizon,
    width,
    device,
):
    action_offset, wall_offset, goal_offset, delta_offset, delta_min, delta_count = offsets
    beams = [{
        "tokens": list(history),
        "types": list(history_types),
        "wall": history[-4] - wall_offset,
        "row": history[-2] - delta_offset + delta_min,
        "col": history[-1] - delta_offset + delta_min,
        "score": 0.0,
        "actions": [],
        "observations": [],
        "done": False,
    }]
    for _ in range(horizon):
        finished = [beam for beam in beams if beam["done"]]
        active = [beam for beam in beams if not beam["done"]]
        if not active:
            break

        action_logits = batched_logits(
            model, [beam["tokens"] for beam in active], [beam["types"] for beam in active], device
        )
        action_prefixes = []
        for beam, logits in zip(active, action_logits):
            scores = torch.log_softmax(
                logits[action_offset:action_offset + len(action_deltas)], dim=-1
            )
            for action in observed_action_indices(beam["wall"], action_deltas):
                action_prefixes.append({
                    **beam,
                    "tokens": beam["tokens"] + [action_offset + action],
                    "types": beam["types"] + [TYPE_ACTION],
                    "score": beam["score"] + float(scores[action]),
                    "actions": beam["actions"] + [action],
                    "selected_action": action,
                })
        action_prefixes = retain_best(action_prefixes, width)
        if not action_prefixes:
            break

        wall_logits = batched_logits(
            model,
            [beam["tokens"] for beam in action_prefixes],
            [beam["types"] for beam in action_prefixes],
            device,
        )
        predicted_walls = [
            int(torch.argmax(logits[wall_offset:wall_offset + 512]).item())
            for logits in wall_logits
        ]
        wall_tokens = [
            beam["tokens"] + [wall_offset + wall]
            for beam, wall in zip(action_prefixes, predicted_walls)
        ]
        wall_types = [beam["types"] + [TYPE_WALLS] for beam in action_prefixes]
        goal_logits = batched_logits(model, wall_tokens, wall_types, device)
        predicted_goals = [
            int(torch.argmax(logits[goal_offset:goal_offset + 10]).item())
            for logits in goal_logits
        ]

        candidates = []
        for beam, tokens, types, wall, visible in zip(
            action_prefixes, wall_tokens, wall_types, predicted_walls, predicted_goals
        ):
            action = beam["selected_action"]
            dr, dc = (int(value) for value in action_deltas[action])
            next_row, next_col = beam["row"] - dr, beam["col"] - dc
            row_token = delta_offset + next_row - delta_min
            col_token = delta_offset + next_col - delta_min
            upper = delta_offset + delta_count
            if not (delta_offset <= row_token < upper and delta_offset <= col_token < upper):
                continue
            observation = [wall_offset + wall, goal_offset + visible, row_token, col_token]
            candidates.append({
                "tokens": tokens + [goal_offset + visible, row_token, col_token],
                "types": types + [TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA],
                "wall": wall,
                "row": next_row,
                "col": next_col,
                "score": beam["score"],
                "actions": beam["actions"],
                "observations": beam["observations"] + [observation],
                "done": next_row == 0 and next_col == 0,
            })
        if not candidates:
            break
        beams = retain_best(finished + candidates, width)

    if not beams or not beams[0]["actions"]:
        raise RuntimeError("No action could be planned from the predicted observation")
    chosen = beams[0]
    return chosen["actions"][0], chosen["observations"][0]


def plan_oracle_context_only(
    model,
    data,
    maze,
    current,
    goal,
    history,
    history_types,
    action_deltas,
    offsets,
    horizon,
    width,
    device,
):
    action_offset, wall_offset, _, _, _, _ = offsets
    beams = [{
        "tokens": list(history),
        "types": list(history_types),
        "wall": history[-4] - wall_offset,
        "cell": current.copy(),
        "score": 0.0,
        "actions": [],
        "observations": [],
        "done": False,
    }]
    for _ in range(horizon):
        finished = [beam for beam in beams if beam["done"]]
        active = [beam for beam in beams if not beam["done"]]
        if not active:
            break
        action_logits = batched_logits(
            model, [beam["tokens"] for beam in active], [beam["types"] for beam in active], device
        )
        action_prefixes = []
        for beam, logits in zip(active, action_logits):
            scores = torch.log_softmax(
                logits[action_offset:action_offset + len(action_deltas)], dim=-1
            )
            for action in observed_action_indices(beam["wall"], action_deltas):
                next_cell = beam["cell"] + action_deltas[action]
                row, col = int(next_cell[0]), int(next_cell[1])
                inside = 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1]
                if not inside or int(maze[row, col]) != 0:
                    continue
                action_prefixes.append({
                    **beam,
                    "tokens": beam["tokens"] + [action_offset + action],
                    "types": beam["types"] + [TYPE_ACTION],
                    "score": beam["score"] + float(scores[action]),
                    "actions": beam["actions"] + [action],
                    "next_cell": next_cell.astype(np.int16),
                })
        action_prefixes = retain_best(action_prefixes, width)
        if not action_prefixes:
            break

        candidates = []
        for beam in action_prefixes:
            observation = observation_tokens(data, maze, beam["next_cell"], goal)
            candidates.append({
                "tokens": beam["tokens"] + observation,
                "types": beam["types"] + [
                    TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA,
                ],
                "wall": observation[0] - wall_offset,
                "cell": beam["next_cell"],
                "score": beam["score"],
                "actions": beam["actions"],
                "observations": beam["observations"] + [observation],
                "done": np.array_equal(beam["next_cell"], goal),
            })
        beams = retain_best(finished + candidates, width)

    if not beams or not beams[0]["actions"]:
        raise RuntimeError("No action could be planned with the oracle observation simulator")
    chosen = beams[0]
    return chosen["actions"][0], chosen["observations"][0]


def rollout(
    model,
    data,
    maze,
    start,
    goal,
    action_deltas,
    max_steps,
    horizon,
    width,
    condition,
    device,
):
    offsets = offsets_from(data)
    action_offset, _, _, _, _, _ = offsets
    bos = int(data["bos_token_id"][0])
    current = start.copy()
    observation = observation_tokens(data, maze, current, goal)
    history = [bos] + observation
    history_types = [1, TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA]
    component_correct = [[], [], [], []]
    complete_correct = []
    invalid = 0
    path = [current.copy()]

    for _ in range(max_steps):
        if condition == "predicted_scored":
            action, predicted = plan_token_level_beam(
                model,
                history,
                history_types,
                action_deltas,
                offsets,
                horizon,
                width,
                device,
            )
        elif condition == "predicted_context_only":
            action, predicted = plan_predicted_context_only(
                model, history, history_types, action_deltas, offsets, horizon, width, device
            )
        else:
            action, predicted = plan_oracle_context_only(
                model,
                data,
                maze,
                current,
                goal,
                history,
                history_types,
                action_deltas,
                offsets,
                horizon,
                width,
                device,
            )

        proposed = current + action_deltas[action]
        row, col = int(proposed[0]), int(proposed[1])
        inside = 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1]
        if inside and int(maze[row, col]) == 0:
            current = proposed.astype(np.int16)
        else:
            invalid += 1
        real_observation = observation_tokens(data, maze, current, goal)
        matches = [int(a) == int(b) for a, b in zip(predicted, real_observation)]
        for values, match in zip(component_correct, matches):
            values.append(match)
        complete_correct.append(all(matches))
        history.extend([action_offset + action] + real_observation)
        history_types.extend([
            TYPE_ACTION, TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA
        ])
        path.append(current.copy())
        if np.array_equal(current, goal):
            break

    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return np.asarray(path), invalid, [mean(values) for values in component_correct], mean(complete_correct)


def summarize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    metrics = (
        "invalid_real_action_rate",
        "first_wall_mask_accuracy",
        "first_visible_goal_accuracy",
        "first_goal_row_accuracy",
        "first_goal_column_accuracy",
        "complete_next_observation_accuracy",
    )
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
        row.update({metric: float(group[metric].mean()) for metric in metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(rows, args) -> None:
    results = pd.DataFrame(rows).sort_values(
        ["condition", "beam_width", "maze_name", "query_index"]
    ).reset_index(drop=True)
    summary = summarize(results, ["condition", "beam_width"])
    per_maze = summarize(results, ["maze_name", "condition", "beam_width"])
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    per_maze.to_csv(args.per_maze_summary_csv, index=False)


def main() -> None:
    args = arguments()
    device = choose_device(args.device)
    data = np.load(args.dataset, allow_pickle=True)
    saved = torch.load(args.checkpoint, map_location=device)
    model = TinyCausalTransformer(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

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
        if len(existing) and not existing["planning_horizon"].eq(args.planning_horizon).all():
            raise ValueError("Existing results use a different planning horizon")
        rows = existing.to_dict("records")
    completed = {
        (int(row["query_index"]), str(row["condition"]), int(row["beam_width"]))
        for row in rows
    }

    print(json.dumps({
        "device": str(device),
        "fold": args.cv_fold,
        "conditions": args.conditions,
        "beam_widths": args.beam_widths,
        "existing_rows": len(rows),
    }, indent=2))

    for maze_name in args.mazes:
        indices = np.flatnonzero((folds == args.cv_fold) & (names == maze_name))
        order = np.lexsort((
            goals[indices, 1], goals[indices, 0],
            starts[indices, 1], starts[indices, 0], lengths[indices],
        ))
        for index in indices[order][:args.queries_per_maze]:
            maze = mazes[int(maze_indices[index])]
            start = starts[index].astype(np.int16)
            goal = goals[index].astype(np.int16)
            for condition in args.conditions:
                for width in args.beam_widths:
                    key = (int(index), condition, int(width))
                    if key in completed:
                        continue
                    before = time.perf_counter()
                    path, invalid, accuracies, complete = rollout(
                        model,
                        data,
                        maze,
                        start,
                        goal,
                        action_deltas,
                        args.max_steps,
                        args.planning_horizon,
                        width,
                        condition,
                        device,
                    )
                    elapsed = time.perf_counter() - before
                    steps = len(path) - 1
                    rows.append({
                        "cv_fold": args.cv_fold,
                        "maze_name": maze_name,
                        "episode_id": int(episode_ids[index]),
                        "query_index": int(index),
                        "condition": condition,
                        "beam_width": int(width),
                        "strategy": f"{condition}_beam_{width}",
                        "steps_until_stop": steps,
                        "compute_seconds": elapsed,
                        "seconds_per_step": elapsed / max(steps, 1),
                        "goal_reached": bool(np.array_equal(path[-1], goal)),
                        "invalid_real_action_count": invalid,
                        "invalid_real_action_rate": invalid / max(steps, 1),
                        "first_wall_mask_accuracy": accuracies[0],
                        "first_visible_goal_accuracy": accuracies[1],
                        "first_goal_row_accuracy": accuracies[2],
                        "first_goal_column_accuracy": accuracies[3],
                        "complete_next_observation_accuracy": complete,
                        "planning_horizon": args.planning_horizon,
                        "observation_source": "oracle" if condition == "oracle_context_only" else "predicted",
                        "observation_probability_in_score": condition == "predicted_scored",
                    })
                    completed.add(key)
        write_outputs(rows, args)
        print(f"Completed {maze_name}: {len(rows)} rows saved", flush=True)

    print(summarize(pd.DataFrame(rows), ["condition", "beam_width"]).to_string(index=False))


if __name__ == "__main__":
    main()
