from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_maze2d_discrete_transformer import TinyCausalTransformer


TYPE_ACTION = 5
TYPE_STATE = 4


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
    return p.parse_args()


def state_token(cell, offset, shape):
    return offset + int(cell[0]) * shape[1] + int(cell[1])


def token_cell(token, offset, shape):
    index = int(token) - offset
    return np.asarray([index // shape[1], index % shape[1]], dtype=np.int16)


def valid_actions(cell, maze, deltas):
    result = []
    for index, delta in enumerate(deltas):
        nxt = cell + delta
        if (0 <= int(nxt[0]) < maze.shape[0] and 0 <= int(nxt[1]) < maze.shape[1]
                and int(maze[int(nxt[0]), int(nxt[1])]) == 0):
            result.append((index, nxt.astype(np.int16)))
    return result


def next_logits(model, batches, types):
    max_context = int(model.pos_encoder.pe.shape[1])
    trimmed_batches, trimmed_types = [], []
    for tokens, token_types in zip(batches, types):
        if len(tokens) > max_context:
            keep_recent = max_context - 3
            tokens = tokens[:3] + tokens[-keep_recent:]
            token_types = token_types[:3] + token_types[-keep_recent:]
        trimmed_batches.append(tokens)
        trimmed_types.append(token_types)
    batches, types = trimmed_batches, trimmed_types
    length = max(map(len, batches))
    x = np.zeros((len(batches), length), dtype=np.int64)
    t = np.zeros_like(x)
    mask = np.zeros_like(x, dtype=np.bool_)
    for row, (tokens, token_types) in enumerate(zip(batches, types)):
        x[row, :len(tokens)] = tokens
        t[row, :len(tokens)] = token_types
        mask[row, :len(tokens)] = True
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(mask))
    return torch.stack([logits[row, len(tokens)-1] for row, tokens in enumerate(batches)])


def plan(model, history, history_types, maze, current, goal, deltas, state_offset,
         action_offset, grid_shape, horizon, width, free_tokens):
    beams = [{"tokens": list(history), "types": list(history_types), "cell": current.copy(),
              "score": 0.0, "actions": [], "states": [], "consistent": [], "done": False}]
    for _ in range(horizon):
        active = [beam for beam in beams if not beam["done"]]
        if not active:
            break
        action_batch = next_logits(model, [b["tokens"] for b in active], [b["types"] for b in active])
        action_expansions = []
        for beam, logits in zip(active, action_batch):
            log_probs = torch.log_softmax(logits[action_offset:action_offset + len(deltas)], dim=-1)
            for action, expected_next in valid_actions(beam["cell"], maze, deltas):
                action_expansions.append((beam, action, expected_next, float(log_probs[action])))
        if not action_expansions:
            break
        state_batches = [b[0]["tokens"] + [action_offset + b[1]] for b in action_expansions]
        state_types = [b[0]["types"] + [TYPE_ACTION] for b in action_expansions]
        state_batch = next_logits(model, state_batches, state_types)
        candidates = []
        for expansion, tokens, types, logits in zip(action_expansions, state_batches, state_types, state_batch):
            beam, action, expected_next, action_score = expansion
            free_logits = logits[free_tokens]
            state_log_probs = torch.log_softmax(free_logits, dim=-1)
            count = min(width, len(free_tokens))
            values, positions = torch.topk(state_log_probs, k=count)
            for value, position in zip(values.tolist(), positions.tolist()):
                predicted_token = int(free_tokens[position])
                predicted_cell = token_cell(predicted_token, state_offset, grid_shape)
                candidates.append({
                    "tokens": tokens + [predicted_token],
                    "types": types + [TYPE_STATE],
                    "cell": predicted_cell,
                    "score": beam["score"] + action_score + float(value),
                    "actions": beam["actions"] + [action],
                    "states": beam["states"] + [predicted_cell],
                    "consistent": beam["consistent"] + [bool(np.array_equal(predicted_cell, expected_next))],
                    "done": bool(np.array_equal(predicted_cell, goal)),
                })
        beams = sorted(candidates, key=lambda item: item["score"], reverse=True)[:width]
        if any(beam["done"] for beam in beams):
            break
    chosen_pool = [beam for beam in beams if beam["done"]] or beams
    chosen = max(chosen_pool, key=lambda item: item["score"])
    return chosen["actions"][0], chosen["states"][0], chosen["consistent"]


def rollout(model, data, maze, maze_token, start, goal, deltas, max_steps, horizon, width, free_tokens):
    state_offset = int(data["state_token_offset"][0])
    action_offset = int(data["action_token_offset"][0])
    shape = tuple(int(v) for v in data["global_grid_shape"])
    current = start.copy()
    history = [int(maze_token), state_token(start, state_offset, shape),
               state_token(goal, state_offset, shape), state_token(start, state_offset, shape)]
    types = [1, 2, 3, TYPE_STATE]
    path = [current.copy()]
    first_state_correct, imagined_consistency = [], []
    for _ in range(max_steps):
        action, predicted_next, consistency = plan(
            model, history, types, maze, current, goal, deltas, state_offset,
            action_offset, shape, horizon, width, free_tokens,
        )
        actual_next = current + deltas[action]
        first_state_correct.append(bool(np.array_equal(predicted_next, actual_next)))
        imagined_consistency.extend(consistency)
        history.extend([action_offset + action, state_token(actual_next, state_offset, shape)])
        types.extend([TYPE_ACTION, TYPE_STATE])
        current = actual_next.astype(np.int16)
        path.append(current.copy())
        if np.array_equal(current, goal):
            break
    return np.asarray(path), float(np.mean(first_state_correct)), float(np.mean(imagined_consistency))


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
                   first_predicted_state_accuracy=float(group.first_predicted_state_accuracy.mean()),
                   imagined_transition_consistency=float(group.imagined_transition_consistency.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = arguments()
    data = np.load(args.dataset, allow_pickle=True)
    saved = torch.load(args.checkpoint, map_location="cpu")
    model = TinyCausalTransformer(**saved["model_config"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    names, folds = np.asarray(data["maze_name_per_episode"]).astype(str), np.asarray(data["fold_ids"])
    starts, goals, lengths = np.asarray(data["start_cells"]), np.asarray(data["goal_cells"]), np.asarray(data["path_lengths"])
    maze_indices, episode_ids = np.asarray(data["maze_indices"]), np.asarray(data["episode_ids"])
    mazes = [np.asarray(x, dtype=np.int8) for x in data["mazes"]]
    maze_tokens = np.asarray(data["maze_token_ids"])
    deltas = np.asarray(data["action_deltas"], dtype=np.int16)
    offset = int(data["state_token_offset"][0])
    shape = tuple(int(v) for v in data["global_grid_shape"])
    rows = []
    for maze_name in args.mazes:
        indices = np.flatnonzero((folds == args.cv_fold) & (names == maze_name))
        order = np.lexsort((goals[indices,1], goals[indices,0], starts[indices,1], starts[indices,0], lengths[indices]))
        for index in indices[order][:args.queries_per_maze]:
            maze_index = int(maze_indices[index])
            maze, start, goal = mazes[maze_index], starts[index].astype(np.int16), goals[index].astype(np.int16)
            free_tokens = torch.tensor([state_token(cell, offset, shape) for cell in np.argwhere(maze == 0)], dtype=torch.long)
            for strategy, width in (("receding_greedy", 1), ("receding_beam_2", 2), ("receding_beam_3", 3)):
                before = time.perf_counter()
                path, state_accuracy, consistency = rollout(
                    model, data, maze, maze_tokens[maze_index], start, goal, deltas,
                    args.max_steps, args.planning_horizon, width, free_tokens,
                )
                elapsed, steps = time.perf_counter() - before, len(path) - 1
                rows.append({"cv_fold": args.cv_fold, "maze_name": maze_name, "episode_id": int(episode_ids[index]),
                             "strategy": strategy, "steps_until_stop": steps, "compute_seconds": elapsed,
                             "seconds_per_step": elapsed / max(steps, 1), "goal_reached": bool(np.array_equal(path[-1], goal)),
                             "first_predicted_state_accuracy": state_accuracy,
                             "imagined_transition_consistency": consistency,
                             "planning_horizon": args.planning_horizon})
    results = pd.DataFrame(rows)
    summary = summarize(results, ["strategy"]).sort_values("strategy")
    per_maze = summarize(results, ["maze_name", "strategy"]).sort_values(["maze_name", "strategy"])
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    per_maze.to_csv(args.per_maze_summary_csv, index=False)
    print(json.dumps({"fold": args.cv_fold, "rows": len(results)}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
