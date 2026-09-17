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
    select_greedy_action,
)
from train_maze2d_discrete_transformer import TinyCausalTransformer


STRATEGIES = ("receding_greedy", "token_level_beam_2", "token_level_beam_3")
METRICS = (
    "invalid_real_action_rate",
    "first_wall_mask_accuracy",
    "first_visible_goal_accuracy",
    "first_goal_row_accuracy",
    "first_goal_column_accuracy",
    "complete_next_observation_accuracy",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cv-fold", type=int, required=True)
    parser.add_argument("--queries-per-maze", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=19)
    parser.add_argument("--planning-horizon", type=int, default=5)
    parser.add_argument("--mazes", nargs="+", required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--per-maze-summary-csv", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
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


def retain_best(candidates: list[dict], width: int) -> list[dict]:
    return sorted(candidates, key=lambda candidate: (-candidate["score"], candidate["tokens"]))[:width]


def plan_token_level_beam(
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

        logits_batch = batched_logits(
            model,
            [beam["tokens"] for beam in active],
            [beam["types"] for beam in active],
            device,
        )
        action_prefixes = []
        for beam, logits in zip(active, logits_batch):
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

        logits_batch = batched_logits(
            model,
            [beam["tokens"] for beam in action_prefixes],
            [beam["types"] for beam in action_prefixes],
            device,
        )
        wall_prefixes = []
        for beam, logits in zip(action_prefixes, logits_batch):
            scores = torch.log_softmax(logits[wall_offset:wall_offset + 512], dim=-1)
            for wall, score in enumerate(scores.tolist()):
                wall_prefixes.append({
                    **beam,
                    "tokens": beam["tokens"] + [wall_offset + wall],
                    "types": beam["types"] + [TYPE_WALLS],
                    "score": beam["score"] + float(score),
                    "predicted_wall": wall,
                })
        wall_prefixes = retain_best(wall_prefixes, width)

        logits_batch = batched_logits(
            model,
            [beam["tokens"] for beam in wall_prefixes],
            [beam["types"] for beam in wall_prefixes],
            device,
        )
        goal_prefixes = []
        for beam, logits in zip(wall_prefixes, logits_batch):
            scores = torch.log_softmax(logits[goal_offset:goal_offset + 10], dim=-1)
            for visible, score in enumerate(scores.tolist()):
                goal_prefixes.append({
                    **beam,
                    "tokens": beam["tokens"] + [goal_offset + visible],
                    "types": beam["types"] + [TYPE_VISIBLE_GOAL],
                    "score": beam["score"] + float(score),
                    "predicted_visible": visible,
                })
        goal_prefixes = retain_best(goal_prefixes, width)

        logits_batch = batched_logits(
            model,
            [beam["tokens"] for beam in goal_prefixes],
            [beam["types"] for beam in goal_prefixes],
            device,
        )
        row_prefixes = []
        upper = delta_offset + delta_count
        for beam, logits in zip(goal_prefixes, logits_batch):
            action = beam["selected_action"]
            dr, dc = (int(value) for value in action_deltas[action])
            next_row = beam["row"] - dr
            next_col = beam["col"] - dc
            row_token = delta_offset + next_row - delta_min
            if not delta_offset <= row_token < upper:
                continue
            scores = torch.log_softmax(logits[delta_offset:upper], dim=-1)
            row_prefixes.append({
                **beam,
                "tokens": beam["tokens"] + [row_token],
                "types": beam["types"] + [TYPE_GOAL_DELTA],
                "score": beam["score"] + float(scores[row_token - delta_offset]),
                "next_row": next_row,
                "next_col": next_col,
                "row_token": row_token,
            })
        row_prefixes = retain_best(row_prefixes, width)
        if not row_prefixes:
            break

        logits_batch = batched_logits(
            model,
            [beam["tokens"] for beam in row_prefixes],
            [beam["types"] for beam in row_prefixes],
            device,
        )
        complete = []
        for beam, logits in zip(row_prefixes, logits_batch):
            col_token = delta_offset + beam["next_col"] - delta_min
            if not delta_offset <= col_token < upper:
                continue
            scores = torch.log_softmax(logits[delta_offset:upper], dim=-1)
            observation = [
                wall_offset + beam["predicted_wall"],
                goal_offset + beam["predicted_visible"],
                beam["row_token"],
                col_token,
            ]
            complete.append({
                "tokens": beam["tokens"] + [col_token],
                "types": beam["types"] + [TYPE_GOAL_DELTA],
                "wall": beam["predicted_wall"],
                "row": beam["next_row"],
                "col": beam["next_col"],
                "score": beam["score"] + float(scores[col_token - delta_offset]),
                "actions": beam["actions"],
                "observations": beam["observations"] + [observation],
                "done": beam["next_row"] == 0 and beam["next_col"] == 0,
            })
        if not complete:
            break
        beams = retain_best(finished + complete, width)

    if not beams or not beams[0]["actions"]:
        raise RuntimeError("No action could be planned from the predicted observation")
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
    device,
    action_only_greedy=False,
):
    offsets = offsets_from(data)
    action_offset, wall_offset, _, _, _, _ = offsets
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
        if action_only_greedy:
            action = select_greedy_action(
                model,
                history,
                history_types,
                action_deltas,
                action_offset,
                wall_offset,
                device,
            )
            predicted = None
        else:
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

        proposed = current + action_deltas[action]
        row, col = int(proposed[0]), int(proposed[1])
        inside = 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1]
        if inside and int(maze[row, col]) == 0:
            current = proposed.astype(np.int16)
        else:
            invalid += 1
        real_observation = observation_tokens(data, maze, current, goal)
        if predicted is not None:
            matches = [int(left) == int(right) for left, right in zip(predicted, real_observation)]
            for values, match in zip(component_correct, matches):
                values.append(match)
            complete_correct.append(all(matches))
        history.extend([action_offset + action] + real_observation)
        history_types.extend([
            TYPE_ACTION,
            TYPE_WALLS,
            TYPE_VISIBLE_GOAL,
            TYPE_GOAL_DELTA,
            TYPE_GOAL_DELTA,
        ])
        path.append(current.copy())
        if np.array_equal(current, goal):
            break

    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return np.asarray(path), invalid, [mean(values) for values in component_correct], mean(complete_correct)


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
        if len(existing):
            beam_rows = existing[existing["strategy"].str.startswith("token_level_beam_")]
            if len(beam_rows) and not beam_rows["planning_horizon"].eq(args.planning_horizon).all():
                raise ValueError("Existing results use a different planning horizon")
        rows = existing.to_dict("records")
    completed = {
        (int(row["query_index"]), str(row["strategy"]))
        for row in rows
    }

    print(json.dumps({
        "device": str(device),
        "fold": args.cv_fold,
        "strategies": args.strategies,
        "existing_rows": len(rows),
    }, indent=2))

    specifications = {
        "receding_greedy": (1, True),
        "token_level_beam_2": (2, False),
        "token_level_beam_3": (3, False),
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
                width, action_only_greedy = specifications[strategy]
                strategy_horizon = 1 if action_only_greedy else args.planning_horizon
                before = time.perf_counter()
                path, invalid, accuracies, complete = rollout(
                    model,
                    data,
                    maze,
                    start,
                    goal,
                    action_deltas,
                    args.max_steps,
                    strategy_horizon,
                    width,
                    device,
                    action_only_greedy=action_only_greedy,
                )
                elapsed = time.perf_counter() - before
                steps = len(path) - 1
                rows.append({
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
                    "first_wall_mask_accuracy": accuracies[0],
                    "first_visible_goal_accuracy": accuracies[1],
                    "first_goal_row_accuracy": accuracies[2],
                    "first_goal_column_accuracy": accuracies[3],
                    "complete_next_observation_accuracy": complete,
                    "planning_horizon": strategy_horizon,
                    "pruning_granularity": "action_only" if action_only_greedy else "token",
                })
                completed.add(key)
        write_outputs(rows, args)
        print(f"Completed {maze_name}: {len(rows)} rows saved", flush=True)

    print(summarize(pd.DataFrame(rows), ["strategy"]).to_string(index=False))


if __name__ == "__main__":
    main()
