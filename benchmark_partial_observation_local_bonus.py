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
from benchmark_partial_observation_token_level_beam import (
    METRICS,
    offsets_from,
    retain_best,
)
from train_maze2d_discrete_transformer import TinyCausalTransformer


STRATEGIES = ("receding_greedy", "token_level_beam_2", "token_level_beam_3")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a decoding-only local-optimality bonus without training a model."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--queries-per-maze", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=19)
    parser.add_argument("--planning-horizon", type=int, default=5)
    parser.add_argument("--mazes", nargs="+", required=True)
    parser.add_argument("--local-bonuses", nargs="+", type=float, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--per-maze-summary-csv", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def locally_optimal_actions(
    wall_mask: int,
    goal_row_delta: int,
    goal_column_delta: int,
    action_deltas: np.ndarray,
) -> set[int]:
    """Return valid actions minimising the two-level local Manhattan criterion."""
    scores: dict[int, tuple[int, int]] = {}
    for action in observed_action_indices(wall_mask, action_deltas):
        dr, dc = (int(value) for value in action_deltas[action])
        candidate_row, candidate_column = 1 + dr, 1 + dc
        first_distance = abs(goal_row_delta - dr) + abs(goal_column_delta - dc)
        neighbour_distances = []
        for next_dr_raw, next_dc_raw in action_deltas:
            next_dr, next_dc = int(next_dr_raw), int(next_dc_raw)
            neighbour_row = candidate_row + next_dr
            neighbour_column = candidate_column + next_dc
            if not (0 <= neighbour_row < 3 and 0 <= neighbour_column < 3):
                continue
            bit = neighbour_row * 3 + neighbour_column
            if ((wall_mask >> bit) & 1) == 0:
                offset_row = neighbour_row - 1
                offset_column = neighbour_column - 1
                neighbour_distances.append(
                    abs(goal_row_delta - offset_row)
                    + abs(goal_column_delta - offset_column)
                )
        second_distance = min(neighbour_distances) if neighbour_distances else first_distance
        scores[action] = (first_distance, second_distance)
    if not scores:
        raise RuntimeError("No valid action is available from the observed wall mask")
    best = min(scores.values())
    return {action for action, score in scores.items() if score == best}


def select_bonus_greedy_action(
    model,
    history,
    history_types,
    action_deltas,
    offsets,
    local_bonus,
    device,
) -> int:
    action_offset, wall_offset, _, delta_offset, delta_min, _ = offsets
    wall_mask = int(history[-4] - wall_offset)
    row_delta = int(history[-2] - delta_offset + delta_min)
    column_delta = int(history[-1] - delta_offset + delta_min)
    valid = observed_action_indices(wall_mask, action_deltas)
    local = locally_optimal_actions(
        wall_mask, row_delta, column_delta, action_deltas
    )
    logits = batched_logits(model, [history], [history_types], device)[0]
    log_probabilities = torch.log_softmax(
        logits[action_offset : action_offset + len(action_deltas)], dim=-1
    )
    return max(
        valid,
        key=lambda action: (
            float(log_probabilities[action]) + local_bonus * int(action in local),
            -action,
        ),
    )


def plan_bonus_token_level_beam(
    model,
    history,
    history_types,
    action_deltas,
    offsets,
    horizon,
    width,
    local_bonus,
    device,
):
    action_offset, wall_offset, goal_offset, delta_offset, delta_min, delta_count = offsets
    beams = [{
        "tokens": list(history),
        "types": list(history_types),
        "wall": int(history[-4] - wall_offset),
        "row": int(history[-2] - delta_offset + delta_min),
        "col": int(history[-1] - delta_offset + delta_min),
        "score": 0.0,
        "actions": [],
        "observations": [],
        "local_actions_selected": 0,
        "actions_scored": 0,
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
            log_probabilities = torch.log_softmax(
                logits[action_offset : action_offset + len(action_deltas)], dim=-1
            )
            local = locally_optimal_actions(
                beam["wall"], beam["row"], beam["col"], action_deltas
            )
            for action in observed_action_indices(beam["wall"], action_deltas):
                is_local = action in local
                action_prefixes.append({
                    **beam,
                    "tokens": beam["tokens"] + [action_offset + action],
                    "types": beam["types"] + [TYPE_ACTION],
                    "score": (
                        beam["score"]
                        + float(log_probabilities[action])
                        + local_bonus * int(is_local)
                    ),
                    "actions": beam["actions"] + [action],
                    "local_actions_selected": beam["local_actions_selected"] + int(is_local),
                    "actions_scored": beam["actions_scored"] + 1,
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
            scores = torch.log_softmax(logits[wall_offset : wall_offset + 512], dim=-1)
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
            scores = torch.log_softmax(logits[goal_offset : goal_offset + 10], dim=-1)
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
            next_row, next_col = beam["row"] - dr, beam["col"] - dc
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
                "local_actions_selected": beam["local_actions_selected"],
                "actions_scored": beam["actions_scored"],
                "done": beam["next_row"] == 0 and beam["next_col"] == 0,
            })
        if not complete:
            break
        beams = retain_best(finished + complete, width)

    if not beams or not beams[0]["actions"]:
        raise RuntimeError("No action could be planned from the predicted observation")
    chosen = beams[0]
    return (
        chosen["actions"][0],
        chosen["observations"][0],
        chosen["local_actions_selected"] / max(chosen["actions_scored"], 1),
    )


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
    local_bonus,
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
    selected_local = []
    invalid = 0
    path = [current.copy()]

    for _ in range(max_steps):
        if action_only_greedy:
            action = select_bonus_greedy_action(
                model,
                history,
                history_types,
                action_deltas,
                offsets,
                local_bonus,
                device,
            )
            predicted = None
            wall = int(history[-4] - wall_offset)
            row = int(history[-2] - offsets[3] + offsets[4])
            col = int(history[-1] - offsets[3] + offsets[4])
            selected_local.append(
                action in locally_optimal_actions(wall, row, col, action_deltas)
            )
        else:
            action, predicted, local_rate = plan_bonus_token_level_beam(
                model,
                history,
                history_types,
                action_deltas,
                offsets,
                horizon,
                width,
                local_bonus,
                device,
            )
            selected_local.append(local_rate)

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
    return (
        np.asarray(path),
        invalid,
        [mean(values) for values in component_correct],
        mean(complete_correct),
        mean(selected_local),
    )


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
            "selected_candidate_local_rate": float(group["selected_candidate_local_rate"].mean()),
            "invalid_real_action_count": int(group["invalid_real_action_count"].sum()),
        })
        row.update({metric: float(group[metric].mean()) for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(rows: list[dict], args: argparse.Namespace) -> None:
    results = pd.DataFrame(rows).sort_values(
        ["local_bonus", "maze_name", "query_index", "strategy"]
    ).reset_index(drop=True)
    summary = summarize(results, ["local_bonus", "strategy"])
    per_maze = summarize(results, ["maze_name", "local_bonus", "strategy"])
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    per_maze.to_csv(args.per_maze_summary_csv, index=False)


def main() -> None:
    args = arguments()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {args.dataset}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Existing trained checkpoint not found: {args.checkpoint}. "
            "This evaluator never trains a model."
        )
    if any(value < 0 for value in args.local_bonuses):
        raise ValueError("Local bonuses must be non-negative")

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
        (int(row["query_index"]), str(row["strategy"]), float(row["local_bonus"]))
        for row in rows
    }

    print(json.dumps({
        "device": str(device),
        "fold": args.cv_fold,
        "checkpoint": str(args.checkpoint),
        "training": False,
        "strategies": args.strategies,
        "local_bonuses": args.local_bonuses,
        "existing_rows": len(rows),
    }, indent=2))

    specifications = {
        "receding_greedy": (1, True),
        "token_level_beam_2": (2, False),
        "token_level_beam_3": (3, False),
    }
    total_requested = 0
    selected_indices: dict[str, np.ndarray] = {}
    for maze_name in args.mazes:
        indices = np.flatnonzero((folds == args.cv_fold) & (names == maze_name))
        order = np.lexsort((
            goals[indices, 1],
            goals[indices, 0],
            starts[indices, 1],
            starts[indices, 0],
            lengths[indices],
        ))
        selected_indices[maze_name] = indices[order][: args.queries_per_maze]
        total_requested += len(selected_indices[maze_name]) * len(args.strategies) * len(args.local_bonuses)
    started = time.perf_counter()
    completed_this_run = 0

    for maze_name in args.mazes:
        for index in selected_indices[maze_name]:
            maze = mazes[int(maze_indices[index])]
            start = starts[index].astype(np.int16)
            goal = goals[index].astype(np.int16)
            for local_bonus in args.local_bonuses:
                for strategy in args.strategies:
                    key = (int(index), strategy, float(local_bonus))
                    if key in completed:
                        continue
                    width, action_only_greedy = specifications[strategy]
                    strategy_horizon = 1 if action_only_greedy else args.planning_horizon
                    before = time.perf_counter()
                    path, invalid, accuracies, complete, selected_local_rate = rollout(
                        model,
                        data,
                        maze,
                        start,
                        goal,
                        action_deltas,
                        args.max_steps,
                        strategy_horizon,
                        width,
                        float(local_bonus),
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
                        "local_bonus": float(local_bonus),
                        "steps_until_stop": steps,
                        "compute_seconds": elapsed,
                        "seconds_per_step": elapsed / max(steps, 1),
                        "goal_reached": bool(np.array_equal(path[-1], goal)),
                        "selected_candidate_local_rate": selected_local_rate,
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
                    completed_this_run += 1
                    if completed_this_run % 100 == 0:
                        write_outputs(rows, args)
                        elapsed_total = time.perf_counter() - started
                        rate = completed_this_run / elapsed_total
                        remaining = max(0, total_requested - len(completed))
                        eta = remaining / rate if rate > 0 else float("nan")
                        print(
                            f"Progress: {len(completed)}/{total_requested} evaluations; "
                            f"elapsed={elapsed_total / 60:.1f} min; ETA={eta / 60:.1f} min",
                            flush=True,
                        )
        write_outputs(rows, args)
        elapsed = time.perf_counter() - started
        done = len(completed)
        rate = completed_this_run / elapsed if elapsed else float("nan")
        remaining = max(0, total_requested - done)
        eta = remaining / rate if rate > 0 else float("nan")
        print(
            f"Completed {maze_name}: {done}/{total_requested} evaluations; "
            f"elapsed={elapsed / 60:.1f} min; ETA={eta / 60:.1f} min",
            flush=True,
        )

    print(summarize(pd.DataFrame(rows), ["local_bonus", "strategy"]).to_string(index=False))


if __name__ == "__main__":
    main()
