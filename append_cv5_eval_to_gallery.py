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

    marker = "## 5-fold cross-validation decoding evaluation"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        print("cv5 evaluation section already present")
        return

    cells = [
        make_markdown_cell(
            "## 5-fold cross-validation decoding evaluation\n\n"
            "This section reports decoding performance under deterministic 5-fold cross-validation over the full dataset. "
            "Each exact `(maze, start, goal)` query is evaluated once as a held-out fold and trained on in the other four folds.\n"
        ),
        make_code_cell(
            """CV5_DATASET_PATH = Path("maze2d_all_mazes_transformer_ready_native_cv5.npz")
CV5_RESULTS_CSV = Path("beam_search_benchmark_native_cv5_results.csv")
CV5_SUMMARY_CSV = Path("beam_search_benchmark_native_cv5_summary.csv")
CV5_PER_MAZE_SUMMARY_CSV = Path("beam_search_benchmark_native_cv5_per_maze_summary.csv")
CV5_FOLD_MEAN_STD_CSV = Path("beam_search_benchmark_native_cv5_fold_mean_std.csv")

cv5_dataset = np.load(CV5_DATASET_PATH, allow_pickle=True)
cv5_results = pd.read_csv(CV5_RESULTS_CSV)
cv5_summary = pd.read_csv(CV5_SUMMARY_CSV)
cv5_per_maze_summary = pd.read_csv(CV5_PER_MAZE_SUMMARY_CSV)
cv5_fold_mean_std = pd.read_csv(CV5_FOLD_MEAN_STD_CSV)

print({
    "dataset": str(CV5_DATASET_PATH),
    "num_folds": int(cv5_dataset["cv_num_folds"][0]),
    "num_queries_total": int(len(cv5_results) // 3),
    "results_rows": int(len(cv5_results)),
})

cv5_summary"""
        ),
        make_code_cell(
            """display(cv5_fold_mean_std)
display(cv5_per_maze_summary)
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
    ax.bar(cv5_summary["strategy"], cv5_summary[metric], color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_title(title)
    ax.set_xlabel("decoding strategy")
    ax.grid(axis="y", alpha=0.3)
    if metric == "completion_rate":
        ax.set_ylim(0.0, 1.05)
    for x_pos, value in enumerate(cv5_summary[metric]):
        ax.text(x_pos, float(value), f"{float(value):.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()
"""
        ),
        make_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(14, 10))
per_maze_plot = cv5_per_maze_summary[cv5_per_maze_summary["maze_name"].isin(["OPEN", "U_MAZE", "SMALL_MAZE", "MEDIUM_MAZE", "LARGE_MAZE"])].copy()
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
    ]

    nb["cells"].extend(cells)
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("appended cv5 evaluation section")


if __name__ == "__main__":
    main()
