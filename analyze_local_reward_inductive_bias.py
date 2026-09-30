from __future__ import annotations

import argparse
import json
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path("maze2d_partial_obs_3x3_joint_obs_action_mixed3_4act_cv5.npz")
DEFAULT_OUTPUT_DIR = Path("analysis_outputs/local_reward_inductive_bias")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether a two-level local Manhattan-distance rule is a useful "
            "inductive bias for the existing mixed-quality partial-observation data."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def local_open_grid(maze: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Return the visible 3x3 cells, with cells outside the maze treated as walls."""
    visible = np.zeros((3, 3), dtype=np.bool_)
    row, column = (int(value) for value in position)
    for local_row in range(3):
        for local_column in range(3):
            maze_row = row + local_row - 1
            maze_column = column + local_column - 1
            if (
                0 <= maze_row < maze.shape[0]
                and 0 <= maze_column < maze.shape[1]
                and int(maze[maze_row, maze_column]) == 0
            ):
                visible[local_row, local_column] = True
    visible[1, 1] = True
    return visible


def local_action_scores(
    maze: np.ndarray,
    position: np.ndarray,
    goal: np.ndarray,
    action_deltas: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """Rank actions by candidate Manhattan distance, then best visible neighbour distance."""
    visible = local_open_grid(maze, position)
    goal_delta = goal.astype(np.int32) - position.astype(np.int32)
    scores: dict[int, tuple[int, int]] = {}
    for action, delta in enumerate(action_deltas):
        dr, dc = (int(value) for value in delta)
        candidate_row, candidate_column = 1 + dr, 1 + dc
        if not visible[candidate_row, candidate_column]:
            continue
        candidate_offset = np.asarray([dr, dc], dtype=np.int32)
        first_distance = int(np.abs(goal_delta - candidate_offset).sum())
        neighbour_distances = []
        for next_dr, next_dc in action_deltas:
            neighbour_row = candidate_row + int(next_dr)
            neighbour_column = candidate_column + int(next_dc)
            if (
                0 <= neighbour_row < 3
                and 0 <= neighbour_column < 3
                and visible[neighbour_row, neighbour_column]
            ):
                neighbour_offset = np.asarray(
                    [neighbour_row - 1, neighbour_column - 1], dtype=np.int32
                )
                neighbour_distances.append(int(np.abs(goal_delta - neighbour_offset).sum()))
        # The centre is always a visible neighbour, so this should never be empty.
        second_distance = min(neighbour_distances) if neighbour_distances else first_distance
        scores[action] = (first_distance, second_distance)
    return scores


def shortest_distances(maze: np.ndarray, goal: np.ndarray) -> np.ndarray:
    distances = np.full(maze.shape, np.iinfo(np.int16).max, dtype=np.int32)
    goal_cell = (int(goal[0]), int(goal[1]))
    distances[goal_cell] = 0
    queue = deque([goal_cell])
    while queue:
        row, column = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_column = row + dr, column + dc
            if (
                0 <= next_row < maze.shape[0]
                and 0 <= next_column < maze.shape[1]
                and int(maze[next_row, next_column]) == 0
                and distances[next_row, next_column] > distances[row, column] + 1
            ):
                distances[next_row, next_column] = distances[row, column] + 1
                queue.append((next_row, next_column))
    return distances


def action_index(delta: np.ndarray, action_deltas: np.ndarray) -> int:
    matches = np.flatnonzero(np.all(action_deltas == delta, axis=1))
    if len(matches) != 1:
        raise ValueError(f"Could not identify action delta {delta.tolist()}")
    return int(matches[0])


def set_audit_record(
    maze_name: str,
    local_actions: set[int],
    global_actions: set[int],
    action_names: list[str],
) -> dict[str, object]:
    intersection = local_actions & global_actions
    return {
        "maze_name": maze_name,
        "local_action_count": len(local_actions),
        "global_action_count": len(global_actions),
        "intersection_count": len(intersection),
        "locally_optimal_actions": ",".join(action_names[index] for index in sorted(local_actions)),
        "globally_optimal_actions": ",".join(action_names[index] for index in sorted(global_actions)),
        "contains_global_optimum": bool(intersection),
        "all_local_choices_globally_optimal": local_actions <= global_actions,
        "local_and_global_sets_equal": local_actions == global_actions,
        "ambiguous_local_tie_contains_bad_action": bool(intersection) and not local_actions <= global_actions,
        "forced_local_global_conflict": not bool(intersection),
        "candidate_local_precision_numerator": len(intersection),
        "candidate_local_precision_denominator": len(local_actions),
        "candidate_global_recall_numerator": len(intersection),
        "candidate_global_recall_denominator": len(global_actions),
    }


def aggregate_contexts(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(columns, keys))
        row.update({
            "decision_contexts": len(group),
            "contains_global_optimum_rate": float(group["contains_global_optimum"].mean()),
            "all_local_choices_globally_optimal_rate": float(
                group["all_local_choices_globally_optimal"].mean()
            ),
            "exact_local_global_set_rate": float(group["local_and_global_sets_equal"].mean()),
            "ambiguous_tie_with_bad_action_rate": float(
                group["ambiguous_local_tie_contains_bad_action"].mean()
            ),
            "forced_conflict_rate": float(group["forced_local_global_conflict"].mean()),
            "mean_number_of_local_optimal_actions": float(group["local_action_count"].mean()),
            "local_action_precision": float(
                group["candidate_local_precision_numerator"].sum()
                / group["candidate_local_precision_denominator"].sum()
            ),
            "global_action_recall": float(
                group["candidate_global_recall_numerator"].sum()
                / group["candidate_global_recall_denominator"].sum()
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_local_vs_training(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compare the local rule with actions actually recorded in the mixed dataset."""
    rows = []
    for keys, group in frame.groupby(columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(columns, keys))
        differences = (
            group["local_optimal_global_precision"]
            - group["training_trajectory_global_optimal_action_rate"]
        )
        row.update({
            "decision_contexts": len(group),
            "training_action_visits": int(group["training_action_visits"].sum()),
            "training_trajectory_contexts": int(
                group["training_trajectory_contexts"].sum()
            ),
            # Macro averages give every maze-position-goal context equal weight.
            "mean_local_optimal_global_precision": float(
                group["local_optimal_global_precision"].mean()
            ),
            "mean_training_global_optimal_action_rate": float(
                group["training_global_optimal_action_rate"].mean()
            ),
            "mean_training_trajectory_global_optimal_action_rate": float(
                group["training_trajectory_global_optimal_action_rate"].mean()
            ),
            "mean_local_minus_training_trajectory_rate": float(differences.mean()),
            "local_higher_context_rate": float((differences > 0).mean()),
            "equal_context_rate": float(np.isclose(differences, 0.0).mean()),
            "training_higher_context_rate": float((differences < 0).mean()),
            # These pooled estimates weight local candidates or recorded visits,
            # respectively, and are included to show sensitivity to weighting.
            "pooled_local_optimal_global_precision": float(
                group["candidate_local_precision_numerator"].sum()
                / group["candidate_local_precision_denominator"].sum()
            ),
            "visit_weighted_training_global_optimal_action_rate": float(
                group["training_globally_optimal_action_visits"].sum()
                / group["training_action_visits"].sum()
            ),
            "trajectory_weighted_training_global_optimal_action_rate": float(
                group["training_trajectories_with_global_first_action"].sum()
                / group["training_trajectory_contexts"].sum()
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def main() -> None:
    args = arguments()
    with np.load(args.dataset, allow_pickle=True) as data:
        paths = [np.asarray(path, dtype=np.int16) for path in data["paths_xy"]]
        goals = np.asarray(data["goal_cells"], dtype=np.int16)
        maze_indices = np.asarray(data["maze_indices"], dtype=np.int32)
        mazes = [np.asarray(maze, dtype=np.int8) for maze in data["mazes"]]
        maze_names = np.asarray(data["maze_name_per_episode"]).astype(str)
        action_deltas = np.asarray(data["action_deltas"], dtype=np.int16)
        action_names = [str(value) for value in data["action_token_names"]]
        query_ids = np.asarray(data["query_ids"], dtype=np.int32)
        rollout_indices = np.asarray(data["rollout_indices"], dtype=np.int32)
        fold_ids = np.asarray(data["fold_ids"], dtype=np.int32)
        reached = np.asarray(data["goal_reached"], dtype=np.bool_)
        qualities = np.asarray(data["trajectory_quality"]).astype(str)
        episode_ids = np.asarray(data["episode_ids"], dtype=np.int32)

    distance_cache: dict[tuple[int, int, int], np.ndarray] = {}
    context_records: dict[tuple[int, int, int, int, int], dict[str, object]] = {}
    trajectory_rows = []

    for episode, path in enumerate(paths):
        maze_index = int(maze_indices[episode])
        maze = mazes[maze_index]
        goal = goals[episode]
        cache_key = (maze_index, int(goal[0]), int(goal[1]))
        if cache_key not in distance_cache:
            distance_cache[cache_key] = shortest_distances(maze, goal)
        distances = distance_cache[cache_key]

        locally_optimal_moves = 0
        locally_suboptimal_moves = 0
        globally_optimal_moves = 0
        shaped_rewards = []
        episode_contexts_seen: set[tuple[int, int, int, int, int]] = set()
        for step, (position, next_position) in enumerate(zip(path[:-1], path[1:])):
            scores = local_action_scores(maze, position, goal, action_deltas)
            best_score = min(scores.values())
            local_actions = {action for action, score in scores.items() if score == best_score}
            actual_action = action_index(next_position - position, action_deltas)
            actual_is_local = actual_action in local_actions
            locally_optimal_moves += int(actual_is_local)
            locally_suboptimal_moves += int(not actual_is_local)

            current_distance = int(distances[tuple(position)])
            global_actions = set()
            for action in scores:
                candidate = position + action_deltas[action]
                if int(distances[tuple(candidate)]) == current_distance - 1:
                    global_actions.add(action)
            globally_optimal_moves += int(actual_action in global_actions)

            context_key = (
                maze_index, int(position[0]), int(position[1]), int(goal[0]), int(goal[1])
            )
            if context_key not in context_records:
                context_records[context_key] = {
                    "maze_index": maze_index,
                    "position_row": int(position[0]),
                    "position_column": int(position[1]),
                    "goal_row": int(goal[0]),
                    "goal_column": int(goal[1]),
                    **set_audit_record(
                        str(maze_names[episode]), local_actions, global_actions, action_names
                    ),
                    "training_action_visits": 0,
                    "training_globally_optimal_action_visits": 0,
                    "training_trajectory_contexts": 0,
                    "training_trajectories_with_global_first_action": 0,
                }
            context_records[context_key]["training_action_visits"] += 1
            context_records[context_key][
                "training_globally_optimal_action_visits"
            ] += int(actual_action in global_actions)
            if context_key not in episode_contexts_seen:
                context_records[context_key]["training_trajectory_contexts"] += 1
                context_records[context_key][
                    "training_trajectories_with_global_first_action"
                ] += int(actual_action in global_actions)
                episode_contexts_seen.add(context_key)

            is_last = step == len(path) - 2
            if is_last and bool(reached[episode]):
                shaped_rewards.append(1.0)
            elif is_last and not bool(reached[episode]):
                shaped_rewards.append(-1.0)
            else:
                shaped_rewards.append(-0.1 if actual_is_local else -0.2)

        steps = len(path) - 1
        trajectory_rows.append({
            "episode_id": int(episode_ids[episode]),
            "query_id": int(query_ids[episode]),
            "rollout_index": int(rollout_indices[episode]),
            "fold_id": int(fold_ids[episode]),
            "maze_name": str(maze_names[episode]),
            "trajectory_quality": str(qualities[episode]),
            "goal_reached": bool(reached[episode]),
            "steps": steps,
            "ordinary_steps": max(0, steps - 1),
            "locally_optimal_moves": locally_optimal_moves,
            "locally_suboptimal_moves": locally_suboptimal_moves,
            "locally_suboptimal_rate": locally_suboptimal_moves / max(steps, 1),
            "globally_optimal_moves": globally_optimal_moves,
            "globally_optimal_rate": globally_optimal_moves / max(steps, 1),
            "local_shaped_return": float(sum(shaped_rewards)),
        })

    trajectories = pd.DataFrame(trajectory_rows)
    contexts = pd.DataFrame(context_records.values())
    contexts["local_optimal_global_precision"] = (
        contexts["candidate_local_precision_numerator"]
        / contexts["candidate_local_precision_denominator"]
    )
    contexts["training_global_optimal_action_rate"] = (
        contexts["training_globally_optimal_action_visits"]
        / contexts["training_action_visits"]
    )
    contexts["training_trajectory_global_optimal_action_rate"] = (
        contexts["training_trajectories_with_global_first_action"]
        / contexts["training_trajectory_contexts"]
    )
    context_summary = pd.concat([
        aggregate_contexts(contexts.assign(scope="ALL"), ["scope"]),
        aggregate_contexts(contexts, ["maze_name"]),
    ], ignore_index=True, sort=False)
    local_vs_training_summary = pd.concat([
        aggregate_local_vs_training(contexts.assign(scope="ALL"), ["scope"]),
        aggregate_local_vs_training(contexts, ["maze_name"]),
    ], ignore_index=True, sort=False)

    pair_rows = []
    for query_id, group in trajectories.groupby("query_id", sort=True):
        if len(group) != 3:
            raise AssertionError(f"Query {query_id} has {len(group)} rather than three trajectories")
        records = list(group.to_dict("records"))
        for left, right in combinations(records, 2):
            row = {
                "query_id": int(query_id),
                "maze_name": left["maze_name"],
                "left_episode_id": left["episode_id"],
                "right_episode_id": right["episode_id"],
                "left_success": left["goal_reached"],
                "right_success": right["goal_reached"],
                "left_steps": left["steps"],
                "right_steps": right["steps"],
                "left_suboptimal": left["locally_suboptimal_moves"],
                "right_suboptimal": right["locally_suboptimal_moves"],
                "left_return": left["local_shaped_return"],
                "right_return": right["local_shaped_return"],
            }
            if left["goal_reached"] != right["goal_reached"]:
                success = left if left["goal_reached"] else right
                failure = right if left["goal_reached"] else left
                row["comparison_type"] = "success_vs_failure"
                row["shaped_ranking"] = (
                    "agreement" if success["local_shaped_return"] > failure["local_shaped_return"]
                    else "tie" if success["local_shaped_return"] == failure["local_shaped_return"]
                    else "reversal"
                )
                row["shorter_has_more_suboptimal"] = np.nan
            elif left["goal_reached"] and left["steps"] != right["steps"]:
                shorter = left if left["steps"] < right["steps"] else right
                longer = right if left["steps"] < right["steps"] else left
                row["comparison_type"] = "successful_different_length"
                row["shaped_ranking"] = (
                    "agreement" if shorter["local_shaped_return"] > longer["local_shaped_return"]
                    else "tie" if shorter["local_shaped_return"] == longer["local_shaped_return"]
                    else "reversal"
                )
                row["shorter_has_more_suboptimal"] = (
                    shorter["locally_suboptimal_moves"] > longer["locally_suboptimal_moves"]
                )
            elif left["goal_reached"] and right["goal_reached"]:
                row["comparison_type"] = "successful_equal_length"
                row["shaped_ranking"] = "not_applicable"
                row["shorter_has_more_suboptimal"] = np.nan
            else:
                row["comparison_type"] = "both_failed"
                row["shaped_ranking"] = "not_applicable"
                row["shorter_has_more_suboptimal"] = np.nan
            pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)

    relevant_pairs = pairs[pairs["comparison_type"].isin(
        ["success_vs_failure", "successful_different_length"]
    )]
    pairwise_summary = (
        relevant_pairs.groupby(["comparison_type", "shaped_ranking"], sort=True)
        .size().rename("pair_count").reset_index()
    )
    pairwise_totals = relevant_pairs.groupby("comparison_type").size().rename("comparison_total")
    pairwise_summary = pairwise_summary.join(pairwise_totals, on="comparison_type")
    pairwise_summary["proportion"] = (
        pairwise_summary["pair_count"] / pairwise_summary["comparison_total"]
    )

    successful_different = pairs[
        pairs["comparison_type"].eq("successful_different_length")
    ]
    successful = trajectories[trajectories["goal_reached"]].copy()
    correlations = pd.DataFrame([{
        "successful_trajectories": len(successful),
        "spearman_steps_vs_suboptimal_count": spearman(
            successful["steps"], successful["locally_suboptimal_moves"]
        ),
        "spearman_steps_vs_suboptimal_rate": spearman(
            successful["steps"], successful["locally_suboptimal_rate"]
        ),
        "spearman_steps_vs_shaped_return": spearman(
            successful["steps"], successful["local_shaped_return"]
        ),
        "different_length_success_pairs": len(successful_different),
        "shorter_has_more_suboptimal_count": int(
            successful_different["shorter_has_more_suboptimal"].fillna(False).sum()
        ),
        "shorter_has_more_suboptimal_rate": float(
            successful_different["shorter_has_more_suboptimal"].mean()
        ),
    }])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(args.output_dir / "trajectory_local_reward_summary.csv", index=False)
    contexts.to_csv(args.output_dir / "unique_context_audit.csv", index=False)
    contexts[contexts["forced_local_global_conflict"]].to_csv(
        args.output_dir / "forced_conflict_examples.csv", index=False
    )
    context_summary.to_csv(args.output_dir / "context_audit_summary.csv", index=False)
    local_vs_training_summary.to_csv(
        args.output_dir / "local_vs_training_global_optimality.csv", index=False
    )
    pairs.to_csv(args.output_dir / "within_query_pairwise_comparisons.csv", index=False)
    pairwise_summary.to_csv(args.output_dir / "pairwise_ranking_summary.csv", index=False)
    correlations.to_csv(args.output_dir / "trajectory_correlation_summary.csv", index=False)

    overall = context_summary[context_summary.get("scope", pd.Series(dtype=str)).eq("ALL")]
    report = {
        "dataset": str(args.dataset),
        "trajectories": len(trajectories),
        "unique_decision_contexts": len(contexts),
        "successful_trajectories": int(trajectories["goal_reached"].sum()),
        "context_audit": overall.to_dict("records")[0] if len(overall) else {},
        "local_vs_training_global_optimality": local_vs_training_summary[
            local_vs_training_summary.get("scope", pd.Series(dtype=str)).eq("ALL")
        ].to_dict("records")[0],
        "trajectory_correlations": correlations.to_dict("records")[0],
        "pairwise_rankings": pairwise_summary.to_dict("records"),
        "output_directory": str(args.output_dir),
    }
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
