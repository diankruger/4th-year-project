# Dataset experiment runner

`run_dataset_experiments.py` runs the existing training and CV decoding scripts from JSON definitions. It does not change the full-dataset notebook or the original runner.

## Dry run

```powershell
& .\.venv\Scripts\python.exe run_dataset_experiments.py `
  --config experiment_configs\cv5_experiments.json `
  --experiment cv5_optimal_4act --fold 0 --dry-run
```

## Run

One development fold:

```powershell
& .\.venv\Scripts\python.exe run_dataset_experiments.py --experiment cv5_optimal_4act --fold 0
```

All configured folds:

```powershell
& .\.venv\Scripts\python.exe run_dataset_experiments.py --experiment cv5_optimal_4act
```

Omit `--experiment` to run every definition. Complete checkpoints/results are reused automatically; `--force` recreates them.

## Add a dataset

Add an item under `experiments`; it inherits the defaults:

```json
{
  "name": "local3x3_seen_layouts",
  "dataset": "datasets/local3x3_seen_layouts_cv5.npz"
}
```

Override only changed settings:

```json
{
  "name": "local3x3_unseen_layouts",
  "dataset": "datasets/local3x3_unseen_layouts_cv5.npz",
  "mazes": ["U_MAZE_EVAL", "MEDIUM_MAZE_EVAL", "LARGE_MAZE_EVAL"],
  "training": {"epochs": 30, "seed": 19},
  "evaluation": {"queries_per_maze": 5000}
}
```

If evaluation queries live in a separate NPZ, keep `dataset` as the training file and add:

```json
"evaluation_dataset": "datasets/local3x3_unseen_layouts_cv5.npz"
```

Both datasets must use compatible token metadata for the selected model. The runner records a separate hash and metadata block for each file.

Nested objects merge with `defaults`. JSON keys such as `batch_size` become `--batch-size`; arrays become multi-value options, and `true` becomes a standalone flag.

Alternative implementations can be selected per experiment:

```json
"scripts": {
  "train": "train_partial_observation_transformer.py",
  "evaluate": "benchmark_partial_observation_decoding.py"
}
```

Training scripts must accept `--dataset`, `--cv-fold`, and `--output`. Evaluation scripts must accept `--dataset`, `--checkpoint`, `--cv-fold`, `--results-csv`, `--summary-csv`, and `--per-maze-summary-csv`.

## Outputs

Artifacts are isolated under `experiment_runs/<name>/`. Each fold contains a checkpoint and three CSVs. The experiment directory contains pooled summaries and `manifest.json`, which records the resolved configuration, dataset hash, interpreter, metadata, and exact commands.
