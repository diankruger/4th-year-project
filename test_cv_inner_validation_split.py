"""Checks for the held-out-test / inner-validation split used in reported runs."""

from pathlib import Path

import numpy as np

from train_maze2d_discrete_transformer import split_cv_training_queries


ROOT = Path(__file__).resolve().parent
DATASETS = (
    "maze2d_all_mazes_transformer_ready_4act_cv5.npz",
    "maze2d_partial_obs_3x3_joint_obs_action_optimal_4act_cv5.npz",
    "maze2d_partial_obs_3x3_joint_obs_action_mixed3_4act_cv5.npz",
)


def query_keys(data, indices):
    return {
        (
            int(data["maze_indices"][i]),
            *map(int, data["start_cells"][i]),
            *map(int, data["goal_cells"][i]),
        )
        for i in indices
    }


def test_all_reported_splits_are_query_disjoint():
    fold_zero_validation_keys = []
    for name in DATASETS:
        with np.load(ROOT / name, allow_pickle=True) as data:
            folds = np.asarray(data["fold_ids"], dtype=np.int32)
            for test_fold in (range(5) if name == DATASETS[0] else (0,)):
                train, validation, metadata = split_cv_training_queries(data, folds, test_fold, 0.1, 7)
                test = np.flatnonzero(folds == test_fold)
                train_keys = query_keys(data, train)
                validation_keys = query_keys(data, validation)
                test_keys = query_keys(data, test)
                assert train_keys.isdisjoint(validation_keys)
                assert train_keys.isdisjoint(test_keys)
                assert validation_keys.isdisjoint(test_keys)
                assert len(train) + len(validation) + len(test) == len(folds)
                assert metadata["validation_queries"] == len(validation_keys)
                if name == DATASETS[2]:
                    assert len(validation) == 3 * len(validation_keys)
                if test_fold == 0:
                    fold_zero_validation_keys.append(validation_keys)
    assert fold_zero_validation_keys[0] == fold_zero_validation_keys[1] == fold_zero_validation_keys[2]


if __name__ == "__main__":
    test_all_reported_splits_are_query_disjoint()
    print("All reported train/validation/test splits are query-disjoint.")
