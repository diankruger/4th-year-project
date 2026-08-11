from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "maze2d_all_mazes_transformer_ready_cv5.npz"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
FOLDS = range(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the 5-fold CV transformer experiments.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--run-tag",
        type=str,
        default="cv5",
        help="Suffix used to name checkpoints and CSV outputs for this run.",
    )
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    subprocess.run(args, cwd=PROJECT_DIR, check=True)


def mean_steps_on_success(df: pd.DataFrame) -> float:
    successful = df.loc[df["goal_reached"], "steps_until_stop"]
    return float(successful.mean()) if len(successful) else float("nan")


def summarize_by_columns(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in results.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "completion_rate": float(group["goal_reached"].mean()),
                "average_steps_to_goal": mean_steps_on_success(group),
                "average_compute_seconds": float(group["compute_seconds"].mean()),
                "average_seconds_per_step": float(group["seconds_per_step"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_summaries = []
    all_per_maze = []

    for fold in FOLDS:
        checkpoint_path = CHECKPOINT_DIR / f"maze2d_multi_maze_transformer_tiny_{args.run_tag}_fold{fold}.pt"
        results_csv = PROJECT_DIR / f"beam_search_benchmark_{args.run_tag}_fold{fold}_results.csv"
        summary_csv = PROJECT_DIR / f"beam_search_benchmark_{args.run_tag}_fold{fold}_summary.csv"
        per_maze_csv = PROJECT_DIR / f"beam_search_benchmark_{args.run_tag}_fold{fold}_per_maze_summary.csv"

        run_command(
            [
                "python",
                "train_maze2d_discrete_transformer.py",
                "--dataset",
                str(args.dataset),
                "--cv-fold",
                str(fold),
                "--epochs",
                "20",
                "--max-train-episodes",
                "1000000",
                "--max-val-episodes",
                "1000000",
                "--batch-size",
                "128",
                "--d-model",
                "64",
                "--nhead",
                "4",
                "--num-layers",
                "2",
                "--ffn-dim",
                "128",
                "--output",
                str(checkpoint_path),
            ]
        )

        run_command(
            [
                "python",
                "benchmark_multi_maze_cv_decoding.py",
                "--dataset",
                str(args.dataset),
                "--checkpoint",
                str(checkpoint_path),
                "--cv-fold",
                str(fold),
                "--queries-per-maze",
                "10000",
                "--results-csv",
                str(results_csv),
                "--summary-csv",
                str(summary_csv),
                "--per-maze-summary-csv",
                str(per_maze_csv),
            ]
        )

        fold_results = pd.read_csv(results_csv)
        fold_summary = pd.read_csv(summary_csv)
        fold_per_maze = pd.read_csv(per_maze_csv)
        all_results.append(fold_results)
        all_summaries.append(fold_summary.assign(cv_fold=fold))
        all_per_maze.append(fold_per_maze.assign(cv_fold=fold))

    pooled_results = pd.concat(all_results, ignore_index=True)
    pooled_fold_summary = pd.concat(all_summaries, ignore_index=True)
    pooled_fold_per_maze = pd.concat(all_per_maze, ignore_index=True)

    cv_summary = summarize_by_columns(pooled_results, ["strategy"]).sort_values("strategy").reset_index(drop=True)
    cv_per_maze_summary = (
        summarize_by_columns(pooled_results, ["maze_name", "strategy"])
        .sort_values(["maze_name", "strategy"])
        .reset_index(drop=True)
    )
    cv_fold_mean_std = (
        pooled_fold_summary.groupby("strategy", as_index=False)
        .agg(
            completion_rate_mean=("completion_rate", "mean"),
            completion_rate_std=("completion_rate", "std"),
            average_steps_to_goal_mean=("average_steps_to_goal", "mean"),
            average_steps_to_goal_std=("average_steps_to_goal", "std"),
            average_compute_seconds_mean=("average_compute_seconds", "mean"),
            average_compute_seconds_std=("average_compute_seconds", "std"),
            average_seconds_per_step_mean=("average_seconds_per_step", "mean"),
            average_seconds_per_step_std=("average_seconds_per_step", "std"),
        )
        .sort_values("strategy")
        .reset_index(drop=True)
    )

    pooled_results.to_csv(PROJECT_DIR / f"beam_search_benchmark_{args.run_tag}_results.csv", index=False)
    pooled_fold_summary.to_csv(PROJECT_DIR / f"beam_search_benchmark_{args.run_tag}_fold_summaries.csv", index=False)
    pooled_fold_per_maze.to_csv(PROJECT_DIR / f"beam_search_benchmark_{args.run_tag}_fold_per_maze_summaries.csv", index=False)
    cv_summary.to_csv(PROJECT_DIR / f"beam_search_benchmark_{args.run_tag}_summary.csv", index=False)
    cv_per_maze_summary.to_csv(PROJECT_DIR / f"beam_search_benchmark_{args.run_tag}_per_maze_summary.csv", index=False)
    cv_fold_mean_std.to_csv(PROJECT_DIR / f"beam_search_benchmark_{args.run_tag}_fold_mean_std.csv", index=False)

    print(cv_summary.to_string(index=False))
    print(cv_fold_mean_std.to_string(index=False))


if __name__ == "__main__":
    main()
