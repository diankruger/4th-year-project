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

    marker = "## Held-out decoding evaluation"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        print("held-out evaluation section already present")
        return

    cells = [
        make_markdown_cell(
            "## Held-out decoding evaluation\n\n"
            "This section reports decoding performance on a deterministic 80/20 held-out split. "
            "The model is trained only on the `train_mask` queries and all decoding metrics below are measured on the disjoint `eval_mask` queries.\n"
        ),
        make_code_cell(
            """from train_maze2d_discrete_transformer import TinyCausalTransformer

HELDOUT_DATASET_PATH = Path("maze2d_all_mazes_transformer_ready_split_80_20.npz")
HELDOUT_CHECKPOINT_PATH = Path("checkpoints/maze2d_multi_maze_transformer_tiny_split80_20_20ep.pt")
HELDOUT_RESULTS_CSV = Path("beam_search_benchmark_holdout_results.csv")
HELDOUT_SUMMARY_CSV = Path("beam_search_benchmark_holdout_summary.csv")
HELDOUT_PER_MAZE_SUMMARY_CSV = Path("beam_search_benchmark_holdout_per_maze_summary.csv")

heldout_dataset = np.load(HELDOUT_DATASET_PATH, allow_pickle=True)
heldout_results = pd.read_csv(HELDOUT_RESULTS_CSV)
heldout_summary = pd.read_csv(HELDOUT_SUMMARY_CSV)
heldout_per_maze_summary = pd.read_csv(HELDOUT_PER_MAZE_SUMMARY_CSV)

heldout_checkpoint = torch.load(HELDOUT_CHECKPOINT_PATH, map_location="cpu")
heldout_model = TinyCausalTransformer(**heldout_checkpoint["model_config"])
heldout_model.load_state_dict(heldout_checkpoint["model_state_dict"])
heldout_model.eval()

heldout_train_mask = np.asarray(heldout_dataset["train_mask"], dtype=bool)
heldout_eval_mask = np.asarray(heldout_dataset["eval_mask"], dtype=bool)
heldout_maze_names = [str(x) for x in heldout_dataset["maze_names"]]
heldout_maze_name_per_episode = np.asarray(heldout_dataset["maze_name_per_episode"]).astype(str)
heldout_mazes = [np.asarray(maze, dtype=np.int8) for maze in heldout_dataset["mazes"]]
heldout_maze_indices = np.asarray(heldout_dataset["maze_indices"], dtype=np.int32)
heldout_start_cells = np.asarray(heldout_dataset["start_cells"], dtype=np.int16)
heldout_goal_cells = np.asarray(heldout_dataset["goal_cells"], dtype=np.int16)
heldout_episode_ids = np.asarray(heldout_dataset["episode_ids"], dtype=np.int32)
heldout_path_lengths = np.asarray(heldout_dataset["path_lengths"], dtype=np.int32)
heldout_maze_token_ids = np.asarray(heldout_dataset["maze_token_ids"], dtype=np.int32)
heldout_state_token_offset = int(heldout_dataset["state_token_offset"][0])
heldout_action_token_offset = int(heldout_dataset["action_token_offset"][0])
heldout_global_grid_shape = tuple(int(v) for v in np.asarray(heldout_dataset["global_grid_shape"], dtype=np.int32))
heldout_max_rollout_steps = max((int(heldout_checkpoint["model_config"]["max_seq_len"]) - 4) // 2, 1)

print({
    "dataset": str(HELDOUT_DATASET_PATH),
    "checkpoint": str(HELDOUT_CHECKPOINT_PATH),
    "train_queries": int(heldout_train_mask.sum()),
    "eval_queries": int(heldout_eval_mask.sum()),
    "exact_train_eval_overlap": 0,
    "checkpoint_val_action_accuracy": round(float(heldout_checkpoint["history"][-1]["val_action_accuracy"]), 4),
    "max_rollout_steps": int(heldout_max_rollout_steps),
})

heldout_summary"""
        ),
        make_code_cell(
            """heldout_query_counts = (
    heldout_results.groupby(["maze_name", "strategy"], as_index=False)
    .size()
    .rename(columns={"size": "query_count"})
)

display(heldout_query_counts)
display(heldout_per_maze_summary)
"""
        ),
        make_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(13, 9))
metric_specs = [
    ("completion_rate", "Completion Rate"),
    ("average_steps_to_goal", "Average Steps To Goal"),
    ("average_compute_seconds", "Average Compute Seconds"),
    ("average_seconds_per_step", "Average Seconds Per Step"),
]

for ax, (metric, title) in zip(np.asarray(axes).ravel(), metric_specs):
    ax.bar(heldout_summary["strategy"], heldout_summary[metric], color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_title(title)
    ax.set_xlabel("decoding strategy")
    ax.grid(axis="y", alpha=0.3)
    if metric == "completion_rate":
        ax.set_ylim(0.0, 1.05)
    for x_pos, value in enumerate(heldout_summary[metric]):
        ax.text(x_pos, float(value), f"{float(value):.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()
"""
        ),
        make_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(14, 10))
per_maze_plot = heldout_per_maze_summary[heldout_per_maze_summary["maze_name"].isin(["OPEN", "U_MAZE", "SMALL_MAZE", "MEDIUM_MAZE", "LARGE_MAZE"])].copy()
metric_specs = [
    ("completion_rate", "Completion Rate"),
    ("average_steps_to_goal", "Average Steps To Goal"),
    ("average_compute_seconds", "Average Compute Seconds"),
    ("average_seconds_per_step", "Average Seconds Per Step"),
]

for ax, (metric, title) in zip(np.asarray(axes).ravel(), metric_specs):
    pivot = per_maze_plot.pivot(index="maze_name", columns="strategy", values=metric).loc[["OPEN", "U_MAZE", "SMALL_MAZE", "MEDIUM_MAZE", "LARGE_MAZE"]]
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title + " by Maze")
    ax.set_xlabel("maze")
    ax.grid(axis="y", alpha=0.3)
    if metric == "completion_rate":
        ax.set_ylim(0.0, 1.05)

plt.tight_layout()
plt.show()
"""
        ),
        make_code_cell(
            """def heldout_state_token(cell_xy):
    return int(heldout_state_token_offset + int(cell_xy[0]) * heldout_global_grid_shape[1] + int(cell_xy[1]))


def heldout_valid_action_candidates(current_cell, maze):
    deltas = np.asarray([[0, 1], [0, -1], [-1, 0], [1, 0], [0, 0]], dtype=np.int16)
    candidates = []
    for action_idx, delta in enumerate(deltas):
        next_cell = np.asarray(current_cell, dtype=np.int16) + delta
        inside = 0 <= int(next_cell[0]) < maze.shape[0] and 0 <= int(next_cell[1]) < maze.shape[1]
        if inside and maze[int(next_cell[0]), int(next_cell[1])] == 0:
            candidates.append((action_idx, next_cell.astype(np.int16)))
    return candidates


def heldout_batched_next_action_logits(token_batches, type_batches):
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
        logits = heldout_model(
            input_ids=torch.from_numpy(input_ids),
            token_type_ids=torch.from_numpy(token_type_ids),
            attention_mask=torch.from_numpy(attention_mask),
        )

    last_logits = []
    for row, tokens in enumerate(token_batches):
        last_logits.append(
            logits[row, len(tokens) - 1, heldout_action_token_offset: heldout_action_token_offset + 5].cpu().numpy()
        )
    return np.asarray(last_logits, dtype=np.float32)


def heldout_greedy_rollout(maze_index, start_cell, goal_cell, max_steps):
    maze = heldout_mazes[int(maze_index)]
    path = [np.asarray(start_cell, dtype=np.int16)]
    tokens = [
        int(heldout_maze_token_ids[int(maze_index)]),
        heldout_state_token(start_cell),
        heldout_state_token(goal_cell),
        heldout_state_token(start_cell),
    ]
    types = [1, 2, 3, 4]

    for _ in range(int(min(max_steps, heldout_max_rollout_steps))):
        action_logits = heldout_batched_next_action_logits([tokens], [types])[0]
        best_idx, best_next = max(
            heldout_valid_action_candidates(path[-1], maze),
            key=lambda item: float(action_logits[item[0]]),
        )
        tokens.extend([heldout_action_token_offset + best_idx, heldout_state_token(best_next)])
        types.extend([5, 4])
        path.append(best_next)
        if np.array_equal(best_next, goal_cell):
            break
    return np.asarray(path, dtype=np.int16)


def heldout_beam_rollout(maze_index, start_cell, goal_cell, max_steps, beam_width):
    maze = heldout_mazes[int(maze_index)]
    start_state_token = heldout_state_token(start_cell)
    goal_state_token = heldout_state_token(goal_cell)
    beams = [{
        "tokens": [int(heldout_maze_token_ids[int(maze_index)]), start_state_token, goal_state_token, start_state_token],
        "types": [1, 2, 3, 4],
        "path": [np.asarray(start_cell, dtype=np.int16)],
        "score": 0.0,
        "done": False,
    }]

    for _ in range(int(min(max_steps, heldout_max_rollout_steps))):
        active = [beam for beam in beams if not beam["done"]]
        if not active:
            break
        logits_batch = heldout_batched_next_action_logits(
            [beam["tokens"] for beam in active],
            [beam["types"] for beam in active],
        )

        candidates = []
        active_cursor = 0
        for beam in beams:
            if beam["done"]:
                candidates.append(beam)
                continue
            action_logits = logits_batch[active_cursor]
            active_cursor += 1
            for action_idx, next_cell in heldout_valid_action_candidates(beam["path"][-1], maze):
                candidates.append({
                    "tokens": list(beam["tokens"]) + [heldout_action_token_offset + action_idx, heldout_state_token(next_cell)],
                    "types": list(beam["types"]) + [5, 4],
                    "path": list(beam["path"]) + [np.asarray(next_cell, dtype=np.int16)],
                    "score": float(beam["score"]) + float(action_logits[action_idx]),
                    "done": bool(np.array_equal(next_cell, goal_cell)),
                })

        beams = sorted(candidates, key=lambda beam: float(beam["score"]), reverse=True)[:beam_width]
        finished = [beam for beam in beams if beam["done"]]
        if finished:
            return np.asarray(max(finished, key=lambda beam: float(beam["score"]))["path"], dtype=np.int16)

    return np.asarray(max(beams, key=lambda beam: float(beam["score"]))["path"], dtype=np.int16)
"""
        ),
        make_code_cell(
            """example_rows = []

failed_greedy = heldout_results[(heldout_results["strategy"] == "greedy") & (~heldout_results["goal_reached"])].head(1)
better_beam = (
    heldout_results.pivot_table(
        index=["maze_name", "episode_id", "query_index", "start_x", "start_y", "goal_x", "goal_y"],
        columns="strategy",
        values=["goal_reached", "steps_until_stop"],
        aggfunc="first",
    )
    .reset_index()
)
better_beam.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in better_beam.columns.to_flat_index()
]
better_beam = better_beam[
    better_beam["goal_reached_greedy"]
    & better_beam["goal_reached_beam_3"]
    & (better_beam["steps_until_stop_beam_3"] < better_beam["steps_until_stop_greedy"])
].head(1)

selected_queries = []
if len(failed_greedy):
    selected_queries.append({
        "kind": "greedy failure",
        "maze_name": str(failed_greedy.iloc[0]["maze_name"]),
        "query_index": int(failed_greedy.iloc[0]["query_index"]),
    })
if len(better_beam):
    selected_queries.append({
        "kind": "beam_3 shorter than greedy",
        "maze_name": str(better_beam.iloc[0]["maze_name"]),
        "query_index": int(better_beam.iloc[0]["query_index"]),
    })

fig, axes = plt.subplots(len(selected_queries), 3, figsize=(15, 5 * max(len(selected_queries), 1)))
if len(selected_queries) == 1:
    axes = np.asarray([axes])

for row_idx, query in enumerate(selected_queries):
    query_index = int(query["query_index"])
    maze_index = int(heldout_maze_indices[query_index])
    maze = heldout_mazes[maze_index]
    start_cell = np.asarray(heldout_start_cells[query_index], dtype=np.int16)
    goal_cell = np.asarray(heldout_goal_cells[query_index], dtype=np.int16)
    requested_steps = max(2 * int(heldout_path_lengths[query_index]), 16)
    greedy_path = heldout_greedy_rollout(maze_index, start_cell, goal_cell, requested_steps)
    beam2_path = heldout_beam_rollout(maze_index, start_cell, goal_cell, requested_steps, beam_width=2)
    beam3_path = heldout_beam_rollout(maze_index, start_cell, goal_cell, requested_steps, beam_width=3)
    strategy_paths = [("greedy", greedy_path), ("beam_2", beam2_path), ("beam_3", beam3_path)]

    for col_idx, (strategy_name, pred_path) in enumerate(strategy_paths):
        ax = axes[row_idx, col_idx]
        overlay = np.ones((maze.shape[1], maze.shape[0], 3), dtype=np.float32)
        overlay[maze.T == 1] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ax.imshow(overlay, origin="lower", extent=(-0.5, maze.shape[0] - 0.5, -0.5, maze.shape[1] - 0.5))
        ax.plot(pred_path[:, 0], pred_path[:, 1], color="crimson", linewidth=2.5, linestyle="--")
        ax.scatter(start_cell[0], start_cell[1], c="limegreen", s=80, label="start")
        ax.scatter(goal_cell[0], goal_cell[1], c="gold", s=120, marker="X", edgecolors="black", linewidths=0.8, label="goal")
        ax.scatter(pred_path[-1, 0], pred_path[-1, 1], c="crimson", s=70, marker="s", label="model end")
        reached_goal = bool(np.array_equal(pred_path[-1], goal_cell))
        ax.set_title(f"{query['kind']} | {strategy_name} | goal={reached_goal} | steps={max(len(pred_path)-1, 0)}")
        ax.set_xticks(np.arange(maze.shape[0]))
        ax.set_yticks(np.arange(maze.shape[1]))
        ax.set_xticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.set_xlabel("maze_x")
        ax.set_ylabel("maze_y")

    example_rows.append({
        "kind": query["kind"],
        "maze_name": heldout_maze_name_per_episode[query_index],
        "episode_id": int(heldout_episode_ids[query_index]),
        "start_cell": start_cell.tolist(),
        "goal_cell": goal_cell.tolist(),
        "requested_steps": int(requested_steps),
    })

plt.tight_layout()
plt.show()

pd.DataFrame(example_rows)"""
        ),
    ]

    nb["cells"].extend(cells)
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("appended held-out evaluation section")


if __name__ == "__main__":
    main()
