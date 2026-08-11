from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiment_configs" / "cv5_experiments.json"
REQUIRED_RESULT_COLUMNS = {
    "strategy", "goal_reached", "steps_until_stop", "compute_seconds", "seconds_per_step"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset-driven training and decoding experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", action="append", default=[], help="Named experiment to run; repeatable.")
    parser.add_argument("--fold", action="append", type=int, default=[], help="Fold to run; repeatable.")
    parser.add_argument("--force", action="store_true", help="Recreate existing artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def load_config(path: Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Only schema_version 1 is supported.")
    if not isinstance(config.get("experiments"), list) or not config["experiments"]:
        raise ValueError("Configuration requires a non-empty 'experiments' list.")
    return config


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def append_cli_options(command: list[str], options: dict[str, Any]) -> None:
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            command.append(flag)
            command.extend(str(item) for item in value)
        else:
            command.extend([flag, str(value)])


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)


def run_command(command: list[str], dry_run: bool) -> None:
    print("$", display_command(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(path: Path, folds: list[int], mazes: list[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = {"input_ids", "fold_ids", "maze_name_per_episode"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"Dataset {path.name} is missing keys: {missing}")
        fold_ids = np.asarray(data["fold_ids"], dtype=np.int32)
        maze_names = np.asarray(data["maze_name_per_episode"]).astype(str)
        available_folds = sorted(int(value) for value in np.unique(fold_ids))
        available_mazes = list(dict.fromkeys(str(value) for value in maze_names))
        absent_folds = sorted(set(folds).difference(available_folds))
        absent_mazes = sorted(set(mazes).difference(available_mazes))
        if absent_folds:
            raise ValueError(f"Folds absent from {path.name}: {absent_folds}")
        if absent_mazes:
            raise ValueError(f"Mazes absent from {path.name}: {absent_mazes}")
        return {
            "path": str(path),
            "sha256": file_sha256(path),
            "episodes": int(data["input_ids"].shape[0]),
            "sequence_length": int(data["input_ids"].shape[1]),
            "available_folds": available_folds,
            "available_mazes": available_mazes,
            "requested_fold_counts": {str(f): int(np.sum(fold_ids == f)) for f in folds},
        }


def mean_steps_on_success(frame: pd.DataFrame) -> float:
    values = frame.loc[frame["goal_reached"].astype(bool), "steps_until_stop"]
    return float(values.mean()) if len(values) else float("nan")


def summarize(results: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in results.groupby(columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(columns, keys)}
        row.update({
            "completion_rate": float(group["goal_reached"].astype(bool).mean()),
            "average_steps_to_goal": mean_steps_on_success(group),
            "average_compute_seconds": float(group["compute_seconds"].mean()),
            "average_seconds_per_step": float(group["seconds_per_step"].mean()),
        })
        for optional_metric in (
            "first_predicted_state_accuracy",
            "imagined_transition_consistency",
        ):
            if optional_metric in group.columns:
                row[optional_metric] = float(group[optional_metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(experiment_dir: Path, paths: list[Path]) -> dict[str, str]:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_RESULT_COLUMNS.difference(frame.columns))
        if missing:
            raise KeyError(f"{path} is missing result columns: {missing}")
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    summary = summarize(pooled, ["strategy"]).sort_values("strategy").reset_index(drop=True)
    outputs: dict[str, Path] = {
        "results": experiment_dir / "results.csv",
        "summary": experiment_dir / "summary.csv",
    }
    pooled.to_csv(outputs["results"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    if "maze_name" in pooled.columns:
        per_maze = summarize(pooled, ["maze_name", "strategy"]).sort_values(["maze_name", "strategy"])
        outputs["per_maze_summary"] = experiment_dir / "per_maze_summary.csv"
        per_maze.to_csv(outputs["per_maze_summary"], index=False)
    if "cv_fold" in pooled.columns:
        fold_summary = summarize(pooled, ["cv_fold", "strategy"]).sort_values(["cv_fold", "strategy"])
        metric_columns = [column for column in fold_summary.columns if column not in {"cv_fold", "strategy"}]
        aggregations = {
            f"{metric}_{stat}": (metric, stat)
            for metric in metric_columns
            for stat in ("mean", "std")
        }
        fold_mean_std = fold_summary.groupby("strategy", as_index=False).agg(**aggregations).sort_values("strategy")
        outputs["fold_summaries"] = experiment_dir / "fold_summaries.csv"
        outputs["fold_mean_std"] = experiment_dir / "fold_mean_std.csv"
        fold_summary.to_csv(outputs["fold_summaries"], index=False)
        fold_mean_std.to_csv(outputs["fold_mean_std"], index=False)
    print(summary.to_string(index=False))
    return {key: str(value) for key, value in outputs.items()}


def validate_experiment(experiment: dict[str, Any]) -> None:
    required = {"name", "dataset", "folds", "mazes", "scripts", "training", "evaluation"}
    missing = sorted(required.difference(experiment))
    if missing:
        raise KeyError(f"Experiment is missing fields: {missing}")
    if not experiment["folds"] or not experiment["mazes"]:
        raise ValueError(f"Experiment {experiment['name']!r} needs folds and mazes.")
    for key in ("train", "evaluate"):
        script = resolve_path(experiment["scripts"][key])
        if not script.is_file():
            raise FileNotFoundError(f"{key} script does not exist: {script}")


def fold_artifacts(root: Path, fold: int) -> dict[str, Path]:
    directory = root / f"fold_{fold}"
    return {
        "directory": directory,
        "checkpoint": directory / "checkpoint.pt",
        "results": directory / "results.csv",
        "summary": directory / "summary.csv",
        "per_maze_summary": directory / "per_maze_summary.csv",
    }


def run_experiment(experiment: dict[str, Any], selected_folds: list[int], force: bool, dry_run: bool) -> None:
    validate_experiment(experiment)
    name = str(experiment["name"])
    folds = [int(f) for f in experiment["folds"] if not selected_folds or int(f) in selected_folds]
    if not folds:
        print(f"Skipping {name}: no matching folds.")
        return
    training_dataset = resolve_path(experiment["dataset"])
    evaluation_dataset = resolve_path(experiment.get("evaluation_dataset", experiment["dataset"]))
    mazes = [str(value) for value in experiment["mazes"]]
    experiment_dir = resolve_path(experiment.get("output_root", "experiment_runs")) / name
    if not dry_run:
        experiment_dir.mkdir(parents=True, exist_ok=True)
    training_dataset_info = inspect_dataset(training_dataset, folds, [])
    evaluation_dataset_info = (
        training_dataset_info
        if evaluation_dataset == training_dataset
        else inspect_dataset(evaluation_dataset, folds, mazes)
    )
    if evaluation_dataset == training_dataset:
        absent_mazes = sorted(set(mazes).difference(training_dataset_info["available_mazes"]))
        if absent_mazes:
            raise ValueError(f"Evaluation mazes are absent from {training_dataset.name}: {absent_mazes}")
    train_script = resolve_path(experiment["scripts"]["train"])
    evaluate_script = resolve_path(experiment["scripts"]["evaluate"])
    result_paths: list[Path] = []
    commands: list[dict[str, Any]] = []
    print(f"\n=== Experiment: {name} ===")
    print("training dataset:")
    print(json.dumps(training_dataset_info, indent=2))
    if evaluation_dataset != training_dataset:
        print("evaluation dataset:")
        print(json.dumps(evaluation_dataset_info, indent=2))

    for fold in folds:
        files = fold_artifacts(experiment_dir, fold)
        if not dry_run:
            files["directory"].mkdir(parents=True, exist_ok=True)
        train = [sys.executable, str(train_script), "--dataset", str(training_dataset), "--cv-fold", str(fold),
                 "--output", str(files["checkpoint"])]
        append_cli_options(train, experiment["training"])
        eval_options = dict(experiment["evaluation"])
        eval_options["mazes"] = mazes
        evaluate = [sys.executable, str(evaluate_script), "--dataset", str(evaluation_dataset), "--checkpoint",
                    str(files["checkpoint"]), "--cv-fold", str(fold), "--results-csv", str(files["results"]),
                    "--summary-csv", str(files["summary"]), "--per-maze-summary-csv",
                    str(files["per_maze_summary"])]
        append_cli_options(evaluate, eval_options)
        do_train = force or not files["checkpoint"].is_file()
        do_evaluate = force or not all(files[k].is_file() for k in ("results", "summary", "per_maze_summary"))
        commands.append({"fold": fold, "train": display_command(train), "evaluate": display_command(evaluate),
                         "training_executed": do_train, "evaluation_executed": do_evaluate})
        if do_train:
            run_command(train, dry_run)
        else:
            print(f"Reusing checkpoint: {files['checkpoint']}")
        if do_evaluate:
            run_command(evaluate, dry_run)
        else:
            print(f"Reusing fold results: {files['results']}")
        result_paths.append(files["results"])

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "resolved_training_dataset": training_dataset_info,
        "resolved_evaluation_dataset": evaluation_dataset_info,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "commands": commands,
        "dry_run": dry_run,
    }
    if dry_run:
        print(json.dumps(manifest, indent=2))
    else:
        manifest["outputs"] = aggregate(experiment_dir, result_paths)
        with (experiment_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"Artifacts: {experiment_dir}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    defaults = config.get("defaults", {})
    experiments = [merge_dicts(defaults, item) for item in config["experiments"]]
    names = [str(item.get("name", "")) for item in experiments]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate experiment names: {duplicates}")
    requested = set(args.experiment)
    unknown = sorted(requested.difference(names))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. Available: {names}")
    for experiment in experiments:
        if not requested or experiment["name"] in requested:
            run_experiment(experiment, args.fold, args.force, args.dry_run)


if __name__ == "__main__":
    main()
