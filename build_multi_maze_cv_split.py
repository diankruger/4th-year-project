from __future__ import annotations

from pathlib import Path

import numpy as np


SOURCE_DATASET = Path("maze2d_all_mazes_transformer_ready.npz")
OUTPUT_DATASET = Path("maze2d_all_mazes_transformer_ready_cv5.npz")
NUM_FOLDS = 5


def main() -> None:
    data = np.load(SOURCE_DATASET, allow_pickle=True)
    files = list(data.files)

    maze_name_per_episode = np.asarray(data["maze_name_per_episode"]).astype("<U32")
    start_cells = np.asarray(data["start_cells"], dtype=np.int16)
    goal_cells = np.asarray(data["goal_cells"], dtype=np.int16)
    path_lengths = np.asarray(data["path_lengths"], dtype=np.int32)

    num_episodes = int(len(maze_name_per_episode))
    fold_ids = np.zeros(num_episodes, dtype=np.int32)

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
        assigned_folds = np.arange(int(maze_sorted_idx.size), dtype=np.int32) % NUM_FOLDS
        fold_ids[maze_sorted_idx] = assigned_folds

        per_maze_rows.append(
            {
                "maze_name": maze_name,
                "total": int(maze_sorted_idx.size),
                "fold_counts": [int(np.sum(assigned_folds == fold)) for fold in range(NUM_FOLDS)],
            }
        )

    out: dict[str, np.ndarray] = {key: data[key] for key in files}
    out["fold_ids"] = fold_ids
    out["cv_num_folds"] = np.asarray([NUM_FOLDS], dtype=np.int32)
    out["cv_split_method"] = np.asarray(["round_robin_lexsorted_per_maze"], dtype="<U64")

    np.savez_compressed(OUTPUT_DATASET, **out)

    print("source_dataset:", SOURCE_DATASET)
    print("output_dataset:", OUTPUT_DATASET)
    print("cv_num_folds:", NUM_FOLDS)
    print("cv_split_method: round_robin_lexsorted_per_maze")
    print("num_episodes:", num_episodes)
    for row in per_maze_rows:
        print(row)


if __name__ == "__main__":
    main()
