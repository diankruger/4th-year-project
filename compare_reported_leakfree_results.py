"""Compare only the experiments reported in the final project, without editing it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
NEW = ROOT / "experiment_runs" / "leakfree_20261026"
COMPARISONS = {
    "Fully observed, optimal data (five folds)": (
        ROOT / "beam_search_benchmark_cv5_4act19_all8_logprob_results.csv",
        NEW / "fully_observed_optimal" / "results.csv",
    ),
    "Partially observed, optimal data (fold 0)": (
        ROOT / "experiment_runs/cv5_partial_obs_token_level_beam_4act/fold_0/results.csv",
        NEW / "partially_observed_optimal" / "results.csv",
    ),
    "Partially observed, mixed-quality data (fold 0)": (
        ROOT / "experiment_runs/fold0_partial_obs_mixed3_token_beam_4act/fold_0/results.csv",
        NEW / "partially_observed_mixed" / "results.csv",
    ),
    "Observation-prediction diagnostic (fold 0)": (
        ROOT / "experiment_runs/cv5_partial_obs_token_level_beam_4act/beam_observation_ablation/results.csv",
        NEW / "partially_observed_optimal" / "observation_diagnostic" / "results.csv",
    ),
    "Local-bonus decoding (fold 0)": (
        ROOT / "experiment_runs/fold0_partial_obs_mixed3_local_bonus_4act/fold_0/results.csv",
        NEW / "partially_observed_mixed" / "local_bonus" / "results.csv",
    ),
}


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    fields = {
        "queries": ("goal_reached", "size"),
        "completion": ("goal_reached", "mean"),
        "steps_all": ("steps_until_stop", "mean"),
        "time_ms": ("compute_seconds", lambda values: 1000 * values.mean()),
        "time_per_step_ms": ("seconds_per_step", lambda values: 1000 * values.mean()),
    }
    if "first_wall_mask_accuracy" in frame:
        fields["wall_agreement"] = ("first_wall_mask_accuracy", "mean")
    return frame.groupby(group_columns, dropna=False).agg(**fields).reset_index()


def compare(title: str, old_path: Path, new_path: Path) -> None:
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    groups = (["local_bonus"] if "local_bonus" in old else []) + ["strategy"]
    keys = ["cv_fold", "query_index", *groups]
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise AssertionError(f"Duplicate evaluation keys in {title}")
    if set(map(tuple, old[keys].to_numpy())) != set(map(tuple, new[keys].to_numpy())):
        raise AssertionError(f"Old and new evaluation queries or strategies differ in {title}")

    old_summary = summarize(old, groups)
    new_summary = summarize(new, groups)
    table = old_summary.merge(new_summary, on=groups, suffixes=("_old", "_new"), validate="one_to_one")
    table["change_pp"] = 100 * (table["completion_new"] - table["completion_old"])
    print(f"\n{title}")
    for row in table.itertuples(index=False):
        label = " / ".join(str(getattr(row, name)) for name in groups)
        message = (
            f"  {label}: completion {row.completion_old:.4f} -> {row.completion_new:.4f} "
            f"({row.change_pp:+.2f} pp); steps/all {row.steps_all_old:.3f} -> "
            f"{row.steps_all_new:.3f}; compute/query {row.time_ms_old:.1f} -> "
            f"{row.time_ms_new:.1f} ms"
        )
        if "wall_agreement_old" in table and pd.notna(getattr(row, "wall_agreement_old")):
            message += (
                f"; wall agreement {row.wall_agreement_old:.4f} -> "
                f"{row.wall_agreement_new:.4f}"
            )
        print(message)

    if "local_bonus" in old or "diagnostic" in title.lower():
        return
    print("  Completion ranking by maze (old -> new):")
    old_maze = old.groupby(["maze_name", "strategy"])["goal_reached"].mean()
    new_maze = new.groupby(["maze_name", "strategy"])["goal_reached"].mean()
    for maze in sorted(old["maze_name"].unique()):
        old_order = old_maze[maze].sort_values(ascending=False, kind="stable")
        new_order = new_maze[maze].sort_values(ascending=False, kind="stable")
        old_text = ", ".join(f"{name} {rate:.3f}" for name, rate in old_order.items())
        new_text = ", ".join(f"{name} {rate:.3f}" for name, rate in new_order.items())
        print(f"    {maze}: {old_text} -> {new_text}")
    for label, frame in (("old", old), ("new", new)):
        pivot = frame.pivot(index=["cv_fold", "query_index"], columns="strategy", values="goal_reached")
        greedy = "greedy" if "greedy" in pivot else "receding_greedy"
        print(f"  Paired decoder comparisons ({label}):")
        for beam in (column for column in pivot if "beam" in column):
            both = pivot[[greedy, beam]].dropna().astype(bool)
            beam_only = int((both[beam] & ~both[greedy]).sum())
            greedy_only = int((both[greedy] & ~both[beam]).sum())
            gap = 100 * (both[beam].mean() - both[greedy].mean())
            print(
                f"    {beam} vs {greedy}: {gap:+.2f} pp; "
                f"beam-only successes {beam_only}, greedy-only successes {greedy_only}"
            )


def main() -> None:
    for title, (old, new) in COMPARISONS.items():
        if not old.is_file() or not new.is_file():
            raise FileNotFoundError(f"Results not complete for {title}: {old}, {new}")
        compare(title, old, new)


if __name__ == "__main__":
    main()
