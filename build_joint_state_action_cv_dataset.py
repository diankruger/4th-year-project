from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("maze2d_all_mazes_transformer_ready_4act_cv5.npz"))
    parser.add_argument("--output", type=Path, default=Path("maze2d_all_mazes_joint_state_action_4act_cv5.npz"))
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=True) as source:
        output = {key: source[key] for key in source.files if key != "labels"}
        input_ids = np.asarray(source["input_ids"], dtype=np.int32)
        token_types = np.asarray(source["token_type_ids"], dtype=np.int8)
        attention = np.asarray(source["attention_mask"], dtype=np.bool_)
        action_labels = np.asarray(source["labels"], dtype=np.int32)
        state_type = int(source["token_type_state"][0])
        action_type = int(source["token_type_action"][0])

    joint_labels = action_labels.copy()
    post_action_state = np.zeros_like(attention)
    post_action_state[:, 1:] = (
        (token_types[:, 1:] == state_type)
        & (token_types[:, :-1] == action_type)
        & attention[:, 1:]
    )
    joint_labels[post_action_state] = input_ids[post_action_state]
    output.update(
        labels=joint_labels,
        action_labels=action_labels,
        joint_labels=joint_labels,
        supervised_action_mask=(action_labels != -100),
        supervised_state_mask=post_action_state,
        supervision=np.asarray(["actions_and_post_action_states"], dtype="<U40"),
        dataset_version=np.asarray(["fully_observed_joint_state_action_v1"], dtype="<U48"),
    )
    np.savez_compressed(args.output, **output)
    print(
        f"wrote {args.output}: episodes={len(input_ids)}, "
        f"action targets={int((action_labels != -100).sum())}, "
        f"state targets={int(post_action_state.sum())}"
    )


if __name__ == "__main__":
    main()
