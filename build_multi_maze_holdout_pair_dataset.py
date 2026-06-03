from __future__ import annotations

from pathlib import Path

import numpy as np


SOURCE_DATASET = Path("maze2d_all_mazes_transformer_ready.npz")
OUTPUT_DATASET = Path("maze2d_all_mazes_transformer_ready_holdout_large_pair.npz")

HOLDOUT_MAZE_NAME = "LARGE_MAZE"
HOLDOUT_START_CELL = np.asarray([7, 10], dtype=np.int16)
HOLDOUT_GOAL_CELL = np.asarray([7, 1], dtype=np.int16)


def main() -> None:
    data = np.load(SOURCE_DATASET, allow_pickle=True)
    files = list(data.files)

    maze_names = [str(x) for x in data["maze_names"]]
    holdout_maze_index = maze_names.index(HOLDOUT_MAZE_NAME)
    maze_indices = np.asarray(data["maze_indices"], dtype=np.int32)
    start_cells = np.asarray(data["start_cells"], dtype=np.int16)
    goal_cells = np.asarray(data["goal_cells"], dtype=np.int16)

    holdout_mask = (
        (maze_indices == holdout_maze_index)
        & np.all(start_cells == HOLDOUT_START_CELL[None, :], axis=1)
        & np.all(goal_cells == HOLDOUT_GOAL_CELL[None, :], axis=1)
    )
    keep_mask = ~holdout_mask

    out: dict[str, np.ndarray] = {}
    per_episode_keys = {
        "input_ids",
        "labels",
        "attention_mask",
        "token_type_ids",
        "sequence_lengths",
        "path_lengths",
        "episode_ids",
        "maze_indices",
        "maze_name_per_episode",
        "grid_shape_per_episode",
        "start_cells",
        "goal_cells",
        "paths_xy",
    }

    for key in files:
        value = data[key]
        if key in per_episode_keys:
            out[key] = value[keep_mask]
        else:
            out[key] = value

    out["episode_ids"] = np.arange(int(keep_mask.sum()), dtype=np.int32)
    out["holdout_maze_name"] = np.asarray([HOLDOUT_MAZE_NAME], dtype="<U32")
    out["holdout_maze_index"] = np.asarray([holdout_maze_index], dtype=np.int32)
    out["holdout_start_cell"] = HOLDOUT_START_CELL.astype(np.int16)
    out["holdout_goal_cell"] = HOLDOUT_GOAL_CELL.astype(np.int16)
    out["holdout_pair_removed_count"] = np.asarray([int(holdout_mask.sum())], dtype=np.int32)

    np.savez_compressed(OUTPUT_DATASET, **out)

    print("source_dataset:", SOURCE_DATASET)
    print("output_dataset:", OUTPUT_DATASET)
    print("holdout_maze_name:", HOLDOUT_MAZE_NAME)
    print("holdout_maze_index:", holdout_maze_index)
    print("holdout_start_cell:", HOLDOUT_START_CELL.tolist())
    print("holdout_goal_cell:", HOLDOUT_GOAL_CELL.tolist())
    print("removed_episodes:", int(holdout_mask.sum()))
    print("remaining_episodes:", int(keep_mask.sum()))


if __name__ == "__main__":
    main()
