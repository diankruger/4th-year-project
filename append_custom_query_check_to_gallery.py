import json
from pathlib import Path


NOTEBOOK_PATH = Path("maze2d_discrete_8x8_gallery.ipynb")


def make_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def make_code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    marker = "## Custom query check"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        print("custom query section already present")
        return

    cells = [
        make_markdown_cell(
            "## Custom query check\n\n"
            "This section checks a user-specified start and goal cell against the same 8x8 maze used above, reports whether the pair exists in the discrete dataset, and only rolls the model out if both cells are valid free cells.\n"
        ),
        make_code_cell(
            """CUSTOM_START_CELL = np.array([6, 6], dtype=np.int16)
CUSTOM_GOAL_CELL = np.array([6, 3], dtype=np.int16)


def cell_is_inside(cell_xy):
    return 0 <= int(cell_xy[0]) < trajectory_grid_shape[0] and 0 <= int(cell_xy[1]) < trajectory_grid_shape[1]


def cell_is_free(cell_xy):
    return cell_is_inside(cell_xy) and trajectory_maze[tuple(np.asarray(cell_xy, dtype=np.int16))] == 0


start_matches = np.all(np.asarray(trajectory_dataset["start_cells"], dtype=np.int16) == CUSTOM_START_CELL[None, :], axis=1)
goal_matches = np.all(np.asarray(trajectory_dataset["goal_cells"], dtype=np.int16) == CUSTOM_GOAL_CELL[None, :], axis=1)
pair_matches = start_matches & goal_matches

all_valid_pairs_seen = (
    len({
        (tuple(map(int, s)), tuple(map(int, g)))
        for s, g in zip(
            np.asarray(trajectory_dataset["start_cells"], dtype=np.int16),
            np.asarray(trajectory_dataset["goal_cells"], dtype=np.int16),
        )
    })
    == int(np.sum(trajectory_maze == 0)) * (int(np.sum(trajectory_maze == 0)) - 1)
)

custom_query_summary = {
    "start_cell": CUSTOM_START_CELL.tolist(),
    "goal_cell": CUSTOM_GOAL_CELL.tolist(),
    "start_inside": bool(cell_is_inside(CUSTOM_START_CELL)),
    "goal_inside": bool(cell_is_inside(CUSTOM_GOAL_CELL)),
    "start_free": bool(cell_is_free(CUSTOM_START_CELL)),
    "goal_free": bool(cell_is_free(CUSTOM_GOAL_CELL)),
    "matching_dataset_episodes": int(np.sum(pair_matches)),
    "all_valid_start_goal_pairs_seen_in_dataset": bool(all_valid_pairs_seen),
}

print(custom_query_summary)
"""
        ),
        make_code_cell(
            """overlay = np.ones((trajectory_maze.shape[1], trajectory_maze.shape[0], 3), dtype=np.float32)
overlay[trajectory_maze.T == 1] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(
    overlay,
    origin="lower",
    extent=(-0.5, trajectory_maze.shape[0] - 0.5, -0.5, trajectory_maze.shape[1] - 0.5),
)

start_ok = bool(cell_is_free(CUSTOM_START_CELL))
goal_ok = bool(cell_is_free(CUSTOM_GOAL_CELL))

ax.scatter(CUSTOM_START_CELL[0], CUSTOM_START_CELL[1], c="limegreen" if start_ok else "orange", s=120, label="custom start")
ax.scatter(CUSTOM_GOAL_CELL[0], CUSTOM_GOAL_CELL[1], c="gold" if goal_ok else "red", s=160, marker="X", edgecolors="black", linewidths=0.8, label="custom goal")

if start_ok and goal_ok:
    custom_pred_path = greedy_rollout_from_start(CUSTOM_START_CELL, CUSTOM_GOAL_CELL)
    ax.plot(custom_pred_path[:, 0], custom_pred_path[:, 1], color="crimson", linewidth=2.0, linestyle="--", label="model rollout")
    ax.scatter(custom_pred_path[-1, 0], custom_pred_path[-1, 1], c="crimson", s=70, marker="s", label="model end")
    reached_goal = bool(np.array_equal(custom_pred_path[-1], CUSTOM_GOAL_CELL))
    title = f"Custom query | goal={reached_goal} | rollout_steps={max(len(custom_pred_path) - 1, 0)}"
else:
    invalid_bits = []
    if not start_ok:
        invalid_bits.append("start is not a free cell")
    if not goal_ok:
        invalid_bits.append("goal is not a free cell")
    title = "Custom query invalid: " + ", ".join(invalid_bits)

ax.set_title(title)
ax.set_xlabel("maze_x / qpos[0]")
ax.set_ylabel("maze_y / qpos[1]")
ax.set_xticks(np.arange(trajectory_grid_shape[0]))
ax.set_yticks(np.arange(trajectory_grid_shape[1]))
ax.set_xticks(np.arange(-0.5, trajectory_grid_shape[0], 1), minor=True)
ax.set_yticks(np.arange(-0.5, trajectory_grid_shape[1], 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)
ax.legend(loc="upper right")
plt.show()
"""
        ),
    ]

    nb["cells"].extend(cells)
    NOTEBOOK_PATH.write_text(json.dumps(nb, separators=(",", ":")), encoding="utf-8")
    print("appended custom query check section")


if __name__ == "__main__":
    main()
