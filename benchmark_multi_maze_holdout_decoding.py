from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_maze2d_discrete_transformer import TinyCausalTransformer


DEFAULT_MAZES = ["OPEN", "U_MAZE", "SMALL_MAZE", "MEDIUM_MAZE", "LARGE_MAZE"]
TOKEN_TYPE_MAZE = 1
TOKEN_TYPE_START = 2
TOKEN_TYPE_GOAL = 3
TOKEN_TYPE_STATE = 4
TOKEN_TYPE_ACTION = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark greedy and beam decoding on held-out Maze2D queries.")
    parser.add_argument("--dataset", type=Path, default=Path("maze2d_all_mazes_transformer_ready_split_80_20.npz"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/maze2d_multi_maze_transformer_tiny_split80_20_20ep.pt"),
    )
    parser.add_argument("--queries-per-maze", type=int, default=20)
    parser.add_argument("--mazes", nargs="*", default=DEFAULT_MAZES)
    parser.add_argument("--results-csv", type=Path, default=Path("beam_search_benchmark_holdout_results.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("beam_search_benchmark_holdout_summary.csv"))
    parser.add_argument(
        "--per-maze-summary-csv",
        type=Path,
        default=Path("beam_search_benchmark_holdout_per_maze_summary.csv"),
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> TinyCausalTransformer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = TinyCausalTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["model_config"]


def state_token(cell_xy: np.ndarray, state_token_offset: int, global_grid_shape: tuple[int, int]) -> int:
    return int(state_token_offset + int(cell_xy[0]) * global_grid_shape[1] + int(cell_xy[1]))


def valid_action_candidates(current_cell: np.ndarray, maze: np.ndarray) -> list[tuple[int, np.ndarray]]:
    candidates: list[tuple[int, np.ndarray]] = []
    deltas = np.asarray([[0, 1], [0, -1], [-1, 0], [1, 0], [0, 0]], dtype=np.int16)
    for action_idx, delta in enumerate(deltas):
        next_cell = current_cell + delta
        inside = 0 <= int(next_cell[0]) < maze.shape[0] and 0 <= int(next_cell[1]) < maze.shape[1]
        if inside and int(maze[int(next_cell[0]), int(next_cell[1])]) == 0:
            candidates.append((action_idx, next_cell.astype(np.int16)))
    return candidates


def batched_next_action_logits(
    model: TinyCausalTransformer,
    token_batches: list[list[int]],
    type_batches: list[list[int]],
    vocab_action_start: int,
    action_count: int,
) -> np.ndarray:
    max_len = max(len(tokens) for tokens in token_batches)
    input_ids = np.zeros((len(token_batches), max_len), dtype=np.int64)
    token_type_ids = np.zeros((len(token_batches), max_len), dtype=np.int64)
    attention_mask = np.zeros((len(token_batches), max_len), dtype=np.bool_)

    for row, (tokens, types) in enumerate(zip(token_batches, type_batches)):
        seq_len = len(tokens)
        input_ids[row, :seq_len] = np.asarray(tokens, dtype=np.int64)
        token_type_ids[row, :seq_len] = np.asarray(types, dtype=np.int64)
        attention_mask[row, :seq_len] = True

    with torch.no_grad():
        logits = model(
            input_ids=torch.from_numpy(input_ids),
            token_type_ids=torch.from_numpy(token_type_ids),
            attention_mask=torch.from_numpy(attention_mask),
        )
    last_logits = []
    for row, tokens in enumerate(token_batches):
        last_logits.append(logits[row, len(tokens) - 1, vocab_action_start : vocab_action_start + action_count].numpy())
    return np.asarray(last_logits, dtype=np.float32)


def greedy_decode_query(
    model: TinyCausalTransformer,
    maze_token_id: int,
    maze: np.ndarray,
    start_cell: np.ndarray,
    goal_cell: np.ndarray,
    state_token_offset: int,
    action_token_offset: int,
    global_grid_shape: tuple[int, int],
    max_steps: int,
) -> np.ndarray:
    path = [np.asarray(start_cell, dtype=np.int16)]
    tokens = [
        int(maze_token_id),
        state_token(start_cell, state_token_offset, global_grid_shape),
        state_token(goal_cell, state_token_offset, global_grid_shape),
        state_token(start_cell, state_token_offset, global_grid_shape),
    ]
    types = [TOKEN_TYPE_MAZE, TOKEN_TYPE_START, TOKEN_TYPE_GOAL, TOKEN_TYPE_STATE]

    for _ in range(max_steps):
        action_logits = batched_next_action_logits(
            model,
            [tokens],
            [types],
            vocab_action_start=action_token_offset,
            action_count=5,
        )[0]
        candidates = valid_action_candidates(path[-1], maze)
        best_idx, best_next = max(candidates, key=lambda item: float(action_logits[item[0]]))
        tokens.extend([action_token_offset + best_idx, state_token(best_next, state_token_offset, global_grid_shape)])
        types.extend([TOKEN_TYPE_ACTION, TOKEN_TYPE_STATE])
        path.append(best_next)
        if np.array_equal(best_next, goal_cell):
            break
    return np.asarray(path, dtype=np.int16)


def beam_decode_query(
    model: TinyCausalTransformer,
    maze_token_id: int,
    maze: np.ndarray,
    start_cell: np.ndarray,
    goal_cell: np.ndarray,
    state_token_offset: int,
    action_token_offset: int,
    global_grid_shape: tuple[int, int],
    max_steps: int,
    beam_width: int,
) -> np.ndarray:
    start_state_token = state_token(start_cell, state_token_offset, global_grid_shape)
    goal_state_token = state_token(goal_cell, state_token_offset, global_grid_shape)
    beams = [
        {
            "tokens": [int(maze_token_id), start_state_token, goal_state_token, start_state_token],
            "types": [TOKEN_TYPE_MAZE, TOKEN_TYPE_START, TOKEN_TYPE_GOAL, TOKEN_TYPE_STATE],
            "path": [np.asarray(start_cell, dtype=np.int16)],
            "score": 0.0,
            "done": False,
        }
    ]

    for _ in range(max_steps):
        active = [beam for beam in beams if not beam["done"]]
        if not active:
            break

        logits_batch = batched_next_action_logits(
            model,
            [beam["tokens"] for beam in active],
            [beam["types"] for beam in active],
            vocab_action_start=action_token_offset,
            action_count=5,
        )

        candidates: list[dict[str, object]] = []
        active_cursor = 0
        for beam in beams:
            if beam["done"]:
                candidates.append(beam)
                continue
            action_logits = logits_batch[active_cursor]
            active_cursor += 1
            current_cell = np.asarray(beam["path"][-1], dtype=np.int16)
            valid_moves = valid_action_candidates(current_cell, maze)
            for action_idx, next_cell in valid_moves:
                next_beam = {
                    "tokens": list(beam["tokens"])
                    + [action_token_offset + action_idx, state_token(next_cell, state_token_offset, global_grid_shape)],
                    "types": list(beam["types"]) + [TOKEN_TYPE_ACTION, TOKEN_TYPE_STATE],
                    "path": list(beam["path"]) + [np.asarray(next_cell, dtype=np.int16)],
                    "score": float(beam["score"]) + float(action_logits[action_idx]),
                    "done": bool(np.array_equal(next_cell, goal_cell)),
                }
                candidates.append(next_beam)

        beams = sorted(candidates, key=lambda beam: float(beam["score"]), reverse=True)[:beam_width]
        if any(bool(beam["done"]) for beam in beams):
            completed = [beam for beam in beams if bool(beam["done"])]
            best_done = max(completed, key=lambda beam: float(beam["score"]))
            return np.asarray(best_done["path"], dtype=np.int16)

    best_beam = max(beams, key=lambda beam: float(beam["score"]))
    return np.asarray(best_beam["path"], dtype=np.int16)


def select_eval_queries(
    dataset: np.lib.npyio.NpzFile,
    maze_name: str,
    query_count: int,
) -> np.ndarray:
    maze_name_per_episode = np.asarray(dataset["maze_name_per_episode"]).astype("<U32")
    eval_mask = np.asarray(dataset["eval_mask"], dtype=np.bool_)
    start_cells = np.asarray(dataset["start_cells"], dtype=np.int16)
    goal_cells = np.asarray(dataset["goal_cells"], dtype=np.int16)
    path_lengths = np.asarray(dataset["path_lengths"], dtype=np.int32)

    idx = np.flatnonzero(eval_mask & (maze_name_per_episode == maze_name))
    order = np.lexsort(
        (
            goal_cells[idx, 1],
            goal_cells[idx, 0],
            start_cells[idx, 1],
            start_cells[idx, 0],
            path_lengths[idx],
        )
    )
    return idx[order][:query_count]


def main() -> None:
    args = parse_args()
    dataset = np.load(args.dataset, allow_pickle=True)
    model, model_config = load_model(args.checkpoint)

    maze_names = [str(x) for x in dataset["maze_names"]]
    mazes = [np.asarray(maze, dtype=np.int8) for maze in dataset["mazes"]]
    maze_name_per_episode = np.asarray(dataset["maze_name_per_episode"]).astype("<U32")
    start_cells = np.asarray(dataset["start_cells"], dtype=np.int16)
    goal_cells = np.asarray(dataset["goal_cells"], dtype=np.int16)
    maze_indices = np.asarray(dataset["maze_indices"], dtype=np.int32)
    episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int32)
    maze_token_ids = np.asarray(dataset["maze_token_ids"], dtype=np.int32)
    state_token_offset = int(dataset["state_token_offset"][0])
    action_token_offset = int(dataset["action_token_offset"][0])
    global_grid_shape = tuple(int(v) for v in np.asarray(dataset["global_grid_shape"], dtype=np.int32))
    transformer_max_rollout_steps = max((int(model_config["max_seq_len"]) - 4) // 2, 1)

    strategies = {
        "greedy": lambda **kwargs: greedy_decode_query(model=model, **kwargs),
        "beam_2": lambda **kwargs: beam_decode_query(model=model, beam_width=2, **kwargs),
        "beam_3": lambda **kwargs: beam_decode_query(model=model, beam_width=3, **kwargs),
    }

    rows: list[dict[str, object]] = []
    for maze_name in args.mazes:
        query_indices = select_eval_queries(dataset, maze_name=maze_name, query_count=args.queries_per_maze)
        for query_idx in query_indices:
            maze_index = int(maze_indices[query_idx])
            maze = mazes[maze_index]
            start_cell = np.asarray(start_cells[query_idx], dtype=np.int16)
            goal_cell = np.asarray(goal_cells[query_idx], dtype=np.int16)
            requested_steps = max(2 * int(dataset["path_lengths"][query_idx]), 16)
            max_steps = min(requested_steps, transformer_max_rollout_steps)

            common_kwargs = {
                "maze_token_id": int(maze_token_ids[maze_index]),
                "maze": maze,
                "start_cell": start_cell,
                "goal_cell": goal_cell,
                "state_token_offset": state_token_offset,
                "action_token_offset": action_token_offset,
                "global_grid_shape": global_grid_shape,
                "max_steps": max_steps,
            }

            for strategy_name, decode_fn in strategies.items():
                t0 = time.perf_counter()
                pred_path = decode_fn(**common_kwargs)
                compute_seconds = float(time.perf_counter() - t0)
                steps_until_stop = int(max(len(pred_path) - 1, 0))
                goal_reached = bool(np.array_equal(pred_path[-1], goal_cell))
                seconds_per_step = compute_seconds / max(steps_until_stop, 1)

                rows.append(
                    {
                        "maze_name": maze_name,
                        "episode_id": int(episode_ids[query_idx]),
                        "query_index": int(query_idx),
                        "start_x": int(start_cell[0]),
                        "start_y": int(start_cell[1]),
                        "goal_x": int(goal_cell[0]),
                        "goal_y": int(goal_cell[1]),
                        "strategy": strategy_name,
                        "requested_steps": requested_steps,
                        "compute_seconds": compute_seconds,
                        "steps_until_stop": steps_until_stop,
                        "seconds_per_step": seconds_per_step,
                        "goal_reached": goal_reached,
                    }
                )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("strategy", as_index=False)
        .agg(
            completion_rate=("goal_reached", "mean"),
            average_steps_to_goal=("steps_until_stop", lambda s: float(s[results.loc[s.index, "goal_reached"]].mean())),
            average_compute_seconds=("compute_seconds", "mean"),
            average_seconds_per_step=("seconds_per_step", "mean"),
        )
        .sort_values("strategy")
        .reset_index(drop=True)
    )
    per_maze_summary = (
        results.groupby(["maze_name", "strategy"], as_index=False)
        .agg(
            completion_rate=("goal_reached", "mean"),
            average_steps_to_goal=("steps_until_stop", lambda s: float(s[results.loc[s.index, "goal_reached"]].mean())),
            average_compute_seconds=("compute_seconds", "mean"),
            average_seconds_per_step=("seconds_per_step", "mean"),
        )
        .sort_values(["maze_name", "strategy"])
        .reset_index(drop=True)
    )

    results.to_csv(args.results_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    per_maze_summary.to_csv(args.per_maze_summary_csv, index=False)

    print(json.dumps({"results_csv": str(args.results_csv), "summary_csv": str(args.summary_csv)}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
