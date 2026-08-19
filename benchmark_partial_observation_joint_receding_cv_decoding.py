from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_partial_observation_cv_dataset import local_encoding
from train_maze2d_discrete_transformer import TinyCausalTransformer


TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_ACTION = 2, 3, 4, 5


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cv-fold", type=int, required=True)
    p.add_argument("--queries-per-maze", type=int, default=10000)
    p.add_argument("--max-steps", type=int, default=19)
    p.add_argument("--planning-horizon", type=int, default=5)
    p.add_argument("--mazes", nargs="*", required=True)
    p.add_argument("--results-csv", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--per-maze-summary-csv", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def trim_complete_groups(tokens: list[int], types: list[int], maximum: int) -> tuple[list[int], list[int]]:
    if len(tokens) <= maximum:
        return tokens, types
    prefix = 5
    group = 5
    available = max(0, maximum - prefix)
    keep = (available // group) * group
    if keep == 0:
        return tokens[:maximum], types[:maximum]
    return tokens[:prefix] + tokens[-keep:], types[:prefix] + types[-keep:]


def batched_logits(model, token_batches, type_batches, device):
    maximum = int(model.pos_encoder.pe.shape[1])
    trimmed = [trim_complete_groups(list(x), list(t), maximum) for x, t in zip(token_batches, type_batches)]
    length = max(len(x[0]) for x in trimmed)
    x = torch.zeros((len(trimmed), length), dtype=torch.long, device=device)
    t = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    positions = []
    for row, (tokens, types) in enumerate(trimmed):
        n = len(tokens)
        x[row, :n] = torch.as_tensor(tokens, device=device)
        t[row, :n] = torch.as_tensor(types, device=device)
        mask[row, :n] = True
        positions.append(n - 1)
    with torch.inference_mode():
        logits = model(x, t, mask)
    return torch.stack([logits[row, pos] for row, pos in enumerate(positions)])


def observed_action_indices(wall_mask: int, action_deltas: np.ndarray) -> list[int]:
    result = []
    for index, (dr, dc) in enumerate(action_deltas):
        bit = (int(dr) + 1) * 3 + (int(dc) + 1)
        if ((wall_mask >> bit) & 1) == 0:
            result.append(index)
    return result


def observation_tokens(data, maze, cell, goal):
    wall_mask, visible, _ = local_encoding(maze, cell, goal)
    delta = goal - cell
    dmin = int(data["goal_delta_min"][0])
    doff = int(data["goal_delta_token_offset"][0])
    return [int(data["wall_token_offset"][0]) + wall_mask,
            int(data["visible_goal_token_offset"][0]) + visible,
            doff + int(delta[0] - dmin), doff + int(delta[1] - dmin)]


def plan_no_oracle_batched(model, history, history_types, action_deltas, offsets, horizon, width, device):
    action_offset, wall_offset, goal_offset, delta_offset, delta_min = offsets
    initial_wall = history[-4] - wall_offset
    initial_row = history[-2] - delta_offset + delta_min
    initial_col = history[-1] - delta_offset + delta_min
    beams = [{"tokens": list(history), "types": list(history_types), "wall": initial_wall,
              "row": initial_row, "col": initial_col, "score": 0.0, "actions": [],
              "observations": [], "consistent": [], "done": False}]
    for _ in range(horizon):
        finished = [b for b in beams if b["done"]]
        active = [b for b in beams if not b["done"]]
        if not active:
            break
        logits_batch = batched_logits(model, [b["tokens"] for b in active], [b["types"] for b in active], device)
        action_expansions = []
        for beam, logits in zip(active, logits_batch):
            scores = torch.log_softmax(logits[action_offset:action_offset + len(action_deltas)], dim=-1)
            valid = observed_action_indices(beam["wall"], action_deltas)
            for action in valid:
                action_expansions.append((beam, action, float(scores[action])))
        if not action_expansions:
            break

        action_tokens = [b[0]["tokens"] + [action_offset + b[1]] for b in action_expansions]
        action_types = [b[0]["types"] + [TYPE_ACTION] for b in action_expansions]
        wall_logits = batched_logits(model, action_tokens, action_types, device)
        wall_expansions = []
        for expansion, tokens, types, logits in zip(action_expansions, action_tokens, action_types, wall_logits):
            count = min(width, 512)
            values, positions = torch.topk(torch.log_softmax(logits[wall_offset:wall_offset + 512], dim=-1), count)
            for value, position in zip(values.tolist(), positions.tolist()):
                wall_expansions.append((expansion, tokens + [wall_offset + position], types + [TYPE_WALLS],
                                        int(position), float(value)))

        wall_tokens = [x[1] for x in wall_expansions]
        wall_types = [x[2] for x in wall_expansions]
        goal_logits = batched_logits(model, wall_tokens, wall_types, device)
        candidates = []
        goal_expansions = []
        for expansion, tokens, types, predicted_wall, wall_score, logits in zip(
                wall_expansions, wall_tokens, wall_types, [x[3] for x in wall_expansions],
                [x[4] for x in wall_expansions], goal_logits):
            count = min(2, 10)
            values, positions = torch.topk(torch.log_softmax(logits[goal_offset:goal_offset + 10], dim=-1), count)
            for value, position in zip(values.tolist(), positions.tolist()):
                goal_expansions.append((expansion[0], expansion[1], expansion[2], predicted_wall,
                                        wall_score, tokens + [goal_offset + position],
                                        types + [TYPE_VISIBLE_GOAL], int(position), float(value)))

        row_logits = batched_logits(model, [x[5] for x in goal_expansions], [x[6] for x in goal_expansions], device)
        row_contexts, row_types, row_scores = [], [], []
        for item, logits in zip(goal_expansions, row_logits):
            beam, action, action_score = item[0]
            dr, dc = (int(v) for v in action_deltas[action])
            next_row, next_col = beam["row"] - dr, beam["col"] - dc
            row_token = delta_offset + next_row - delta_min
            if not delta_offset <= row_token < delta_offset + 23:
                continue
            score = float(torch.log_softmax(logits[delta_offset:delta_offset + 23], dim=-1)[row_token - delta_offset])
            row_contexts.append(item[5] + [row_token])
            row_types.append(item[6] + [TYPE_GOAL_DELTA])
            row_scores.append((item, next_row, next_col, score))
        if not row_contexts:
            break
        col_logits = batched_logits(model, row_contexts, row_types, device)
        for context, types, packed, logits in zip(row_contexts, row_types, row_scores, col_logits):
            item, next_row, next_col, row_score = packed
            beam, action, action_score = item[0]
            col_token = delta_offset + next_col - delta_min
            if not delta_offset <= col_token < delta_offset + 23:
                continue
            col_score = float(torch.log_softmax(logits[delta_offset:delta_offset + 23], dim=-1)[col_token - delta_offset])
            observation = [item[3] + wall_offset, item[7] + goal_offset,
                           delta_offset + next_row - delta_min, col_token]
            candidates.append({"tokens": context + [col_token], "types": types + [TYPE_GOAL_DELTA],
                               "wall": item[3], "row": next_row, "col": next_col,
                               "score": beam["score"] + action_score + item[4] + item[8] + row_score + col_score,
                               "actions": beam["actions"] + [action],
                               "observations": beam["observations"] + [observation],
                               "consistent": beam["consistent"] + [True],
                               "done": next_row == 0 and next_col == 0})
        if not candidates:
            break
        beams = sorted(finished + candidates, key=lambda b: (-b["score"], b["tokens"]))[:width]
    if not beams or not beams[0]["actions"]:
        raise RuntimeError("No action could be planned from the observed wall mask")
    chosen = beams[0]
    return chosen["actions"][0], chosen["observations"][0], chosen["consistent"]


def rollout(model, data, maze, start, goal, action_deltas, max_steps, horizon, width, device):
    action_offset = int(data["action_token_offset"][0]); wall_offset = int(data["wall_token_offset"][0])
    goal_offset = int(data["visible_goal_token_offset"][0]); delta_offset = int(data["goal_delta_token_offset"][0])
    delta_min = int(data["goal_delta_min"][0]); bos = int(data["bos_token_id"][0])
    current = start.copy()
    obs = observation_tokens(data, maze, current, goal)
    history, types = [bos] + obs, [1, TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA]
    component_correct = [[], [], [], []]
    complete_correct, consistency, invalid = [], [], 0
    path = [current.copy()]
    for _ in range(max_steps):
        action, predicted, imagined_consistency = plan_no_oracle_batched(
            model, history, types, action_deltas,
            (action_offset, wall_offset, goal_offset, delta_offset, delta_min), horizon, width, device)
        proposed = current + action_deltas[action]
        inside = 0 <= int(proposed[0]) < maze.shape[0] and 0 <= int(proposed[1]) < maze.shape[1]
        if inside and int(maze[int(proposed[0]), int(proposed[1])]) == 0:
            current = proposed.astype(np.int16)
        else:
            invalid += 1
        real_obs = observation_tokens(data, maze, current, goal)
        matches = [int(a) == int(b) for a, b in zip(predicted, real_obs)]
        for bucket, match in zip(component_correct, matches):
            bucket.append(match)
        complete_correct.append(all(matches)); consistency.extend(imagined_consistency)
        history.extend([action_offset + action] + real_obs)
        types.extend([TYPE_ACTION, TYPE_WALLS, TYPE_VISIBLE_GOAL, TYPE_GOAL_DELTA, TYPE_GOAL_DELTA])
        path.append(current.copy())
        if np.array_equal(current, goal):
            break
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return np.asarray(path), invalid, [mean(x) for x in component_correct], mean(complete_correct), mean(consistency)


METRICS = ("invalid_real_action_rate", "first_wall_mask_accuracy", "first_visible_goal_accuracy",
           "first_goal_row_accuracy", "first_goal_column_accuracy", "complete_next_observation_accuracy",
           "imagined_displacement_consistency")


def summarize(frame, columns):
    rows = []
    for keys, group in frame.groupby(columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        successful = group.loc[group.goal_reached, "steps_until_stop"]
        row = dict(zip(columns, keys))
        row.update(completion_rate=float(group.goal_reached.mean()),
                   average_steps_to_goal=float(successful.mean()) if len(successful) else float("nan"),
                   average_compute_seconds=float(group.compute_seconds.mean()),
                   average_seconds_per_step=float(group.seconds_per_step.mean()),
                   invalid_real_action_count=int(group.invalid_real_action_count.sum()))
        row.update({metric: float(group[metric].mean()) for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = arguments(); device = choose_device(args.device)
    data = np.load(args.dataset, allow_pickle=True)
    saved = torch.load(args.checkpoint, map_location=device)
    model = TinyCausalTransformer(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    names = np.asarray(data["maze_name_per_episode"]).astype(str); folds = np.asarray(data["fold_ids"])
    starts = np.asarray(data["start_cells"]); goals = np.asarray(data["goal_cells"]); lengths = np.asarray(data["path_lengths"])
    maze_indices = np.asarray(data["maze_indices"]); episode_ids = np.asarray(data["episode_ids"])
    mazes = [np.asarray(x, dtype=np.int8) for x in data["mazes"]]
    deltas = np.asarray(data["action_deltas"], dtype=np.int16); rows = []
    print(json.dumps({"device": str(device), "fold": args.cv_fold}, indent=2))
    for maze_name in args.mazes:
        indices = np.flatnonzero((folds == args.cv_fold) & (names == maze_name))
        order = np.lexsort((goals[indices, 1], goals[indices, 0], starts[indices, 1], starts[indices, 0], lengths[indices]))
        for index in indices[order][:args.queries_per_maze]:
            maze = mazes[int(maze_indices[index])]; start = starts[index].astype(np.int16); goal = goals[index].astype(np.int16)
            for strategy, width in (("receding_greedy", 1), ("receding_beam_2", 2), ("receding_beam_3", 3)):
                before = time.perf_counter()
                path, invalid, accuracies, complete, consistency = rollout(
                    model, data, maze, start, goal, deltas, args.max_steps, args.planning_horizon, width, device)
                elapsed, steps = time.perf_counter() - before, len(path) - 1
                rows.append({"cv_fold": args.cv_fold, "maze_name": maze_name, "episode_id": int(episode_ids[index]),
                             "query_index": int(index), "strategy": strategy, "steps_until_stop": steps,
                             "compute_seconds": elapsed, "seconds_per_step": elapsed / max(steps, 1),
                             "goal_reached": bool(np.array_equal(path[-1], goal)),
                             "invalid_real_action_count": invalid, "invalid_real_action_rate": invalid / max(steps, 1),
                             "first_wall_mask_accuracy": accuracies[0], "first_visible_goal_accuracy": accuracies[1],
                             "first_goal_row_accuracy": accuracies[2], "first_goal_column_accuracy": accuracies[3],
                             "complete_next_observation_accuracy": complete,
                             "imagined_displacement_consistency": consistency, "planning_horizon": args.planning_horizon})
    results = pd.DataFrame(rows); summary = summarize(results, ["strategy"]).sort_values("strategy")
    per_maze = summarize(results, ["maze_name", "strategy"]).sort_values(["maze_name", "strategy"])
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False); summary.to_csv(args.summary_csv, index=False)
    per_maze.to_csv(args.per_maze_summary_csv, index=False)
    print(json.dumps({"fold": args.cv_fold, "rows": len(results), "device": str(device)}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
