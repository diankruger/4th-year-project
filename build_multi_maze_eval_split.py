from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SOURCE_DATASET = Path("maze2d_all_mazes_transformer_ready.npz")
OUTPUT_DATASET = Path("maze2d_all_mazes_transformer_ready_split_80_20.npz")
EVAL_FRACTION = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an 80/20 evaluation split for a transformer-ready maze dataset.")
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--output-dataset", type=Path, default=OUTPUT_DATASET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.source_dataset, allow_pickle=True)
    files = list(data.files)

    maze_name_per_episode = np.asarray(data["maze_name_per_episode"]).astype("<U32")
    start_cells = np.asarray(data["start_cells"], dtype=np.int16)
    goal_cells = np.asarray(data["goal_cells"], dtype=np.int16)
    path_lengths = np.asarray(data["path_lengths"], dtype=np.int32)

    num_episodes = int(len(maze_name_per_episode))
    eval_mask = np.zeros(num_episodes, dtype=np.bool_)

    unique_mazes = list(dict.fromkeys(str(name) for name in maze_name_per_episode))
    per_maze_rows: list[dict[str, object]] = []

    for maze_name in unique_mazes:
        maze_idx = np.flatnonzero(maze_name_per_episode == maze_name)
        sort_order = np.lexsort(
            (
                goal_cells[maze_idx, 1],
                goal_cells[maze_idx, 0],
                start_cells[maze_idx, 1],
                start_cells[maze_idx, 0],
                path_lengths[maze_idx],
            )
        )
        maze_sorted_idx = maze_idx[sort_order]
        maze_count = int(maze_sorted_idx.size)
        eval_count = max(1, int(round(EVAL_FRACTION * maze_count)))
        eval_pick = maze_sorted_idx[np.arange(0, maze_count, 5, dtype=np.int32)[:eval_count]]
        eval_mask[eval_pick] = True

        per_maze_rows.append(
            {
                "maze_name": maze_name,
                "total": maze_count,
                "eval": int(eval_pick.size),
                "train": int(maze_count - eval_pick.size),
            }
        )

    train_mask = ~eval_mask

    out: dict[str, np.ndarray] = {key: data[key] for key in files}
    out["train_mask"] = train_mask
    out["eval_mask"] = eval_mask
    out["split_method"] = np.asarray(["every_5th_lexsorted_per_maze"], dtype="<U64")
    out["eval_fraction_requested"] = np.asarray([EVAL_FRACTION], dtype=np.float32)
    out["train_episode_count"] = np.asarray([int(train_mask.sum())], dtype=np.int32)
    out["eval_episode_count"] = np.asarray([int(eval_mask.sum())], dtype=np.int32)

    np.savez_compressed(args.output_dataset, **out)

    print("source_dataset:", args.source_dataset)
    print("output_dataset:", args.output_dataset)
    print("split_method: every_5th_lexsorted_per_maze")
    print("train_episodes:", int(train_mask.sum()))
    print("eval_episodes:", int(eval_mask.sum()))
    for row in per_maze_rows:
        print(row)


if __name__ == "__main__":
    main()
