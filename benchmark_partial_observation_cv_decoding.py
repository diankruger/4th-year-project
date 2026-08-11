from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_maze2d_discrete_transformer import TinyCausalTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cv-fold", type=int, required=True)
    p.add_argument("--queries-per-maze", type=int, default=10000)
    p.add_argument("--max-steps", type=int, default=19)
    p.add_argument("--mazes", nargs="*", required=True)
    p.add_argument("--results-csv", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--per-maze-summary-csv", type=Path, required=True)
    return p.parse_args()


def encode_observation(data, maze: np.ndarray, cell: np.ndarray, goal: np.ndarray) -> tuple[list[int], list[int]]:
    wall_offset = int(data["wall_token_offset"][0])
    goal_offset = int(data["visible_goal_token_offset"][0])
    delta_offset = int(data["goal_delta_token_offset"][0])
    delta_min = int(data["goal_delta_min"][0])
    mask, goal_position, bit = 0, 0, 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            row, col = int(cell[0] + dr), int(cell[1] + dc)
            free = 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1] and int(maze[row, col]) == 0
            if not free:
                mask |= 1 << bit
            if free and row == int(goal[0]) and col == int(goal[1]):
                goal_position = bit + 1
            bit += 1
    delta = goal - cell
    return [wall_offset + mask, goal_offset + goal_position,
            delta_offset + int(delta[0] - delta_min), delta_offset + int(delta[1] - delta_min)], [2, 3, 4, 4]


def valid_actions(cell: np.ndarray, maze: np.ndarray, deltas: np.ndarray):
    result = []
    for index, delta in enumerate(deltas):
        nxt = cell + delta
        inside = 0 <= int(nxt[0]) < maze.shape[0] and 0 <= int(nxt[1]) < maze.shape[1]
        if inside and int(maze[int(nxt[0]), int(nxt[1])]) == 0:
            result.append((index, nxt.astype(np.int16)))
    return result


def action_logits(model, batches, type_batches, offset, count):
    max_len = max(map(len, batches))
    x = np.zeros((len(batches), max_len), dtype=np.int64)
    t = np.zeros_like(x)
    m = np.zeros_like(x, dtype=np.bool_)
    for row, (tokens, types) in enumerate(zip(batches, type_batches)):
        x[row, :len(tokens)], t[row, :len(types)], m[row, :len(tokens)] = tokens, types, True
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(m))
    return np.asarray([torch.log_softmax(logits[i, len(seq)-1, offset:offset+count], dim=-1).numpy()
                       for i, seq in enumerate(batches)])


def decode(model, data, maze, start, goal, deltas, max_steps, width):
    bos = int(data["bos_token_id"][0])
    offset = int(data["action_token_offset"][0])
    obs, obs_types = encode_observation(data, maze, start, goal)
    beams = [{"tokens": [bos] + obs, "types": [1] + obs_types, "path": [start.copy()], "score": 0.0, "done": False}]
    for _ in range(max_steps):
        active = [b for b in beams if not b["done"]]
        if not active:
            break
        logits = action_logits(model, [b["tokens"] for b in active], [b["types"] for b in active], offset, len(deltas))
        candidates, cursor = [], 0
        for beam in beams:
            if beam["done"]:
                candidates.append(beam)
                continue
            scores = logits[cursor]
            cursor += 1
            for action, nxt in valid_actions(beam["path"][-1], maze, deltas):
                next_obs, next_types = encode_observation(data, maze, nxt, goal)
                candidates.append({
                    "tokens": beam["tokens"] + [offset + action] + next_obs,
                    "types": beam["types"] + [5] + next_types,
                    "path": beam["path"] + [nxt],
                    "score": beam["score"] + float(scores[action]),
                    "done": bool(np.array_equal(nxt, goal)),
                })
        beams = sorted(candidates, key=lambda b: b["score"], reverse=True)[:width]
        finished = [b for b in beams if b["done"]]
        if finished:
            return np.asarray(max(finished, key=lambda b: b["score"])["path"])
    return np.asarray(max(beams, key=lambda b: b["score"])["path"])


def summarize(frame, columns):
    rows = []
    for keys, group in frame.groupby(columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        success = group.loc[group.goal_reached, "steps_until_stop"]
        row = dict(zip(columns, keys))
        row.update(completion_rate=float(group.goal_reached.mean()),
                   average_steps_to_goal=float(success.mean()) if len(success) else float("nan"),
                   average_compute_seconds=float(group.compute_seconds.mean()),
                   average_seconds_per_step=float(group.seconds_per_step.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data = np.load(args.dataset, allow_pickle=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = TinyCausalTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    names = np.asarray(data["maze_name_per_episode"]).astype(str)
    folds = np.asarray(data["fold_ids"])
    starts, goals = np.asarray(data["start_cells"]), np.asarray(data["goal_cells"])
    lengths = np.asarray(data["path_lengths"])
    maze_indices, episode_ids = np.asarray(data["maze_indices"]), np.asarray(data["episode_ids"])
    mazes = [np.asarray(x, dtype=np.int8) for x in data["mazes"]]
    deltas = np.asarray(data["action_deltas"], dtype=np.int16)
    rows = []
    for maze_name in args.mazes:
        indices = np.flatnonzero((folds == args.cv_fold) & (names == maze_name))
        order = np.lexsort((goals[indices,1], goals[indices,0], starts[indices,1], starts[indices,0], lengths[indices]))
        for index in indices[order][:args.queries_per_maze]:
            maze, start, goal = mazes[int(maze_indices[index])], starts[index].astype(np.int16), goals[index].astype(np.int16)
            for strategy, width in (("greedy", 1), ("beam_2", 2), ("beam_3", 3)):
                before = time.perf_counter()
                path = decode(model, data, maze, start, goal, deltas, args.max_steps, width)
                elapsed = time.perf_counter() - before
                steps = len(path) - 1
                rows.append({"cv_fold": args.cv_fold, "maze_name": maze_name, "episode_id": int(episode_ids[index]),
                             "query_index": int(index), "start_x": int(start[0]), "start_y": int(start[1]),
                             "goal_x": int(goal[0]), "goal_y": int(goal[1]), "strategy": strategy,
                             "requested_steps": args.max_steps, "compute_seconds": elapsed,
                             "steps_until_stop": steps, "seconds_per_step": elapsed / max(steps, 1),
                             "goal_reached": bool(np.array_equal(path[-1], goal)),
                             "beam_uses_oracle_hypothetical_observations": width > 1})
    results = pd.DataFrame(rows)
    summary = summarize(results, ["strategy"]).sort_values("strategy")
    per_maze = summarize(results, ["maze_name", "strategy"]).sort_values(["maze_name", "strategy"])
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    per_maze.to_csv(args.per_maze_summary_csv, index=False)
    print(json.dumps({"cv_fold": args.cv_fold, "queries": len(results)}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
