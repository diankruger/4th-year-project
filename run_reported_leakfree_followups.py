"""Run only the two reported decoding follow-ups after leak-free model runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "experiment_runs" / "leakfree_20261026"
MAZES = (
    "OPEN", "U_MAZE", "U_MAZE_EVAL", "SMALL_MAZE",
    "MEDIUM_MAZE", "MEDIUM_MAZE_EVAL", "LARGE_MAZE", "LARGE_MAZE_EVAL",
)
EVALUATION_DATASET = ROOT / "maze2d_partial_obs_3x3_joint_obs_action_optimal_4act_cv5.npz"


def followup_commands() -> list[list[str]]:
    optimal = RUNS / "partially_observed_optimal" / "fold_0"
    mixed = RUNS / "partially_observed_mixed" / "fold_0"
    diagnostic = RUNS / "partially_observed_optimal" / "observation_diagnostic"
    bonus = RUNS / "partially_observed_mixed" / "local_bonus"
    common = [
        "--dataset", str(EVALUATION_DATASET),
        "--cv-fold", "0",
        "--queries-per-maze", "10000",
        "--max-steps", "19",
        "--planning-horizon", "5",
        "--mazes", *MAZES,
        "--device", "auto",
        "--resume",
    ]
    ablation = [
        sys.executable, str(ROOT / "benchmark_partial_observation_beam_ablation.py"),
        "--checkpoint", str(optimal / "checkpoint.pt"),
        *common,
        "--beam-widths", "2", "3",
        "--conditions", "predicted_context_only", "oracle_context_only",
        "--results-csv", str(diagnostic / "results.csv"),
        "--summary-csv", str(diagnostic / "summary.csv"),
        "--per-maze-summary-csv", str(diagnostic / "per_maze_summary.csv"),
    ]
    local_bonus = [
        sys.executable, str(ROOT / "benchmark_partial_observation_local_bonus.py"),
        "--checkpoint", str(mixed / "checkpoint.pt"),
        *common,
        "--local-bonuses", "0", "0.25", "0.5", "1", "2",
        "--strategies", "receding_greedy", "token_level_beam_2", "token_level_beam_3",
        "--results-csv", str(bonus / "results.csv"),
        "--summary-csv", str(bonus / "summary.csv"),
        "--per-maze-summary-csv", str(bonus / "per_maze_summary.csv"),
    ]
    return [ablation, local_bonus]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-base", action="store_true")
    args = parser.parse_args()
    commands = followup_commands()
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return

    base_manifest = RUNS / "partially_observed_mixed" / "manifest.json"
    if args.wait_for_base:
        deadline = time.monotonic() + 12 * 60 * 60
        while not base_manifest.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Base experiments did not finish within 12 hours: {base_manifest}")
            time.sleep(30)
    if not base_manifest.is_file():
        raise FileNotFoundError(f"Base experiments have not finished: {base_manifest}")

    for command in commands:
        output_directory = Path(command[command.index("--results-csv") + 1]).parent
        output_directory.mkdir(parents=True, exist_ok=True)
        print("Running", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        print("Finished", output_directory, flush=True)

    comparison_path = RUNS / "old_vs_leakfree_comparison.txt"
    with comparison_path.open("w", encoding="utf-8") as output:
        subprocess.run(
            [sys.executable, str(ROOT / "compare_reported_leakfree_results.py")],
            cwd=ROOT,
            stdout=output,
            check=True,
        )
    print("Finished comparison", comparison_path, flush=True)


if __name__ == "__main__":
    main()
