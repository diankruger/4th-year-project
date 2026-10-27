"""Print report table rows from the leak-free raw evaluation results."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent / "experiment_runs" / "leakfree_20261026"
MAZES = (
    "OPEN", "U_MAZE", "U_MAZE_EVAL", "SMALL_MAZE",
    "MEDIUM_MAZE", "MEDIUM_MAZE_EVAL", "LARGE_MAZE", "LARGE_MAZE_EVAL",
)


def report_rows(name: str, frame: pd.DataFrame, strategies: tuple[str, ...], metric: str, decimals: int) -> None:
    print(f"\n{name}")
    for maze in MAZES:
        cells = []
        for strategy in strategies:
            values = frame.loc[(frame.maze_name == maze) & (frame.strategy == strategy)]
            if values.empty:
                raise ValueError(f"Missing {maze}/{strategy} for {name}")
            if metric == "completion":
                value = values.goal_reached.mean()
            elif metric == "steps":
                value = values.steps_until_stop.mean()
            elif metric == "compute_ms":
                value = 1000 * values.compute_seconds.mean()
            elif metric == "per_step_ms":
                value = 1000 * values.seconds_per_step.mean()
            else:
                raise ValueError(metric)
            cells.append(f"{value:.{decimals}f}")
        print(maze, " & ", " & ".join(cells))


def table_rows(source: str, label: str) -> list[list[str]]:
    block = source.split(f"\\label{{{label}}}", 1)[1].split("\\end{tabular}", 1)[0]
    rows = []
    for line in block.splitlines():
        if "&" not in line or not line.rstrip().endswith(r"\\"):
            continue
        line = line.replace(r"\texttt{", "").replace("}", "").replace(r"\_", "_")
        rows.append([cell.strip().strip("$") for cell in line.removesuffix(r"\\").split("&")])
    return rows


def check_maze_table(source: str, label: str, frame: pd.DataFrame,
                     strategies: tuple[str, ...], metric: str, decimals: int) -> None:
    rows = {row[0].upper(): row for row in table_rows(source, label)[1:]}
    for maze in MAZES:
        actual = rows[maze][-len(strategies):]
        expected = []
        for strategy in strategies:
            values = frame.loc[(frame.maze_name == maze) & (frame.strategy == strategy)]
            value = {
                "completion": values.goal_reached.mean(),
                "steps": values.steps_until_stop.mean(),
                "compute_ms": 1000 * values.compute_seconds.mean(),
                "per_step_ms": 1000 * values.seconds_per_step.mean(),
            }[metric]
            expected.append(f"{value:.{decimals}f}")
        assert actual == expected, (label, maze, actual, expected)


def main() -> None:
    full = pd.read_csv(ROOT / "fully_observed_optimal" / "results.csv")
    optimal = pd.read_csv(ROOT / "partially_observed_optimal" / "results.csv")
    mixed = pd.read_csv(ROOT / "partially_observed_mixed" / "results.csv")
    diagnostic = pd.read_csv(ROOT / "partially_observed_optimal" / "observation_diagnostic" / "results.csv")
    bonus = pd.read_csv(ROOT / "partially_observed_mixed" / "local_bonus" / "results.csv")
    assert len(full) == 3 * 5976
    assert len(optimal) == len(mixed) == 3 * 1196
    assert len(diagnostic) == 4 * 1196
    assert len(bonus) == 15 * 1196

    full_strategies = ("greedy", "beam_2", "beam_3")
    partial_strategies = ("receding_greedy", "token_level_beam_2", "token_level_beam_3")
    for metric, decimals in (("completion", 4), ("steps", 4), ("compute_ms", 3), ("per_step_ms", 3)):
        report_rows("FULL " + metric, full, full_strategies, metric, decimals)
        report_rows("OPTIMAL " + metric, optimal, partial_strategies, metric, decimals)

    print("\nDIAGNOSTIC (strategy, completion, steps, wall-mask agreement)")
    for frame, strategies in (
        (optimal, partial_strategies[1:]),
        (diagnostic, (
            "predicted_context_only_beam_2", "predicted_context_only_beam_3",
            "oracle_context_only_beam_2", "oracle_context_only_beam_3",
        )),
    ):
        for strategy in strategies:
            rows = frame.loc[frame.strategy == strategy]
            print(strategy, f"{rows.goal_reached.mean():.4f}",
                  f"{rows.steps_until_stop.mean():.4f}",
                  f"{rows.first_wall_mask_accuracy.mean():.4f}")

    print("\nMIXED (strategy, completion, steps, wall-mask agreement)")
    for strategy in partial_strategies:
        rows = mixed.loc[mixed.strategy == strategy]
        print(strategy, f"{rows.goal_reached.mean():.4f}",
              f"{rows.steps_until_stop.mean():.4f}",
              f"{rows.first_wall_mask_accuracy.mean():.4f}")

    print("\nBONUS (strength, strategy, completion, steps)")
    for strength in (0, 0.25, 0.5, 1, 2):
        for strategy in partial_strategies:
            rows = bonus.loc[(bonus.local_bonus == strength) & (bonus.strategy == strategy)]
            print(strength, strategy, f"{rows.goal_reached.mean():.4f}",
                  f"{rows.steps_until_stop.mean():.4f}")

    source = (Path(__file__).resolve().parent / "Planning as autoregressive sequence modelling B Decoding.tex").read_text(encoding="utf-8")
    for metric, decimals, full_label, optimal_label in (
        ("completion", 4, "tab:completion_rate_all8", "tab:partial_completion_rate"),
        ("steps", 4, "tab:avg_steps_all8", "tab:partial_avg_steps"),
        ("compute_ms", 3, "tab:compute_time_all8", "tab:partial_compute_time"),
        ("per_step_ms", 3, "tab:time_per_step_all8", "tab:partial_time_per_step"),
    ):
        check_maze_table(source, full_label, full, full_strategies, metric, decimals)
        check_maze_table(source, optimal_label, optimal, partial_strategies, metric, decimals)

    diagnostic_rows = table_rows(source, "tab:partial_observation_ablation")[1:]
    diagnostic_frames = [optimal, optimal, diagnostic, diagnostic, diagnostic, diagnostic]
    diagnostic_strategies = (
        "token_level_beam_2", "token_level_beam_3",
        "predicted_context_only_beam_2", "predicted_context_only_beam_3",
        "oracle_context_only_beam_2", "oracle_context_only_beam_3",
    )
    for row, frame, strategy in zip(diagnostic_rows, diagnostic_frames, diagnostic_strategies, strict=True):
        values = frame.loc[frame.strategy == strategy]
        expected = [f"{values.goal_reached.mean():.4f}",
                    f"{values.steps_until_stop.mean():.4f}",
                    f"{values.first_wall_mask_accuracy.mean():.4f}"]
        assert row[-3:] == expected, (strategy, row[-3:], expected)

    mixed_rows = table_rows(source, "tab:partial_mixed_quality_results")[1:]
    for row, strategy in zip(mixed_rows, partial_strategies, strict=True):
        values = mixed.loc[mixed.strategy == strategy]
        expected = [f"{values.goal_reached.mean():.4f}",
                    f"{values.steps_until_stop.mean():.4f}",
                    "--" if strategy == "receding_greedy" else f"{values.first_wall_mask_accuracy.mean():.4f}"]
        assert row[-3:] == expected, (strategy, row[-3:], expected)

    bonus_rows = table_rows(source, "tab:partial_rule_guided_results")[1:]
    for row, (strength, strategy) in zip(
        bonus_rows,
        ((strength, strategy) for strength in (0, 0.25, 0.5, 1, 2) for strategy in partial_strategies),
        strict=True,
    ):
        values = bonus.loc[(bonus.local_bonus == strength) & (bonus.strategy == strategy)]
        expected = [f"{values.goal_reached.mean():.4f}", f"{values.steps_until_stop.mean():.4f}"]
        assert row[-2:] == expected, (strength, strategy, row[-2:], expected)
    print("\nAll eleven result tables match the raw leak-free evaluation CSVs.")


if __name__ == "__main__":
    main()
