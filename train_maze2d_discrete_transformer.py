from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MazeTokenDataset(Dataset):
    def __init__(self, npz_path: Path, indices: np.ndarray) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.input_ids = torch.from_numpy(data["input_ids"][indices].astype(np.int64))
        self.labels = torch.from_numpy(data["labels"][indices].astype(np.int64))
        self.attention_mask = torch.from_numpy(data["attention_mask"][indices].astype(np.bool_))
        self.token_type_ids = torch.from_numpy(data["token_type_ids"][indices].astype(np.int64))
        self.sequence_lengths = torch.from_numpy(data["sequence_lengths"][indices].astype(np.int64))
        self.episode_ids = torch.from_numpy(data["episode_ids"][indices].astype(np.int64))
        if "value_targets" in data.files:
            self.value_targets = torch.from_numpy(data["value_targets"][indices].astype(np.float32))
        else:
            self.value_targets = torch.full(self.input_ids.shape, float("nan"), dtype=torch.float32)

        self.vocab_size = int(data["vocab_size"][0])
        self.num_token_types = int(self.token_type_ids.max().item()) + 1
        self.pad_token_id = int(data["pad_token_id"][0])
        self.state_token_offset = int(data["state_token_offset"][0])
        self.action_token_offset = int(data["action_token_offset"][0])
        grid_key = "grid_shape" if "grid_shape" in data.files else "global_grid_shape"
        self.grid_shape = tuple(int(v) for v in data[grid_key])
        self.action_token_names = [str(v) for v in data["action_token_names"]]
        self.action_deltas = np.asarray(data["action_deltas"], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "sequence_lengths": self.sequence_lengths[idx],
            "episode_id": self.episode_ids[idx],
            "value_targets": self.value_targets[idx],
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_token_types: int = 6,
        value_head: bool = False,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.type_embed = nn.Embedding(num_token_types, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.has_value_head = bool(value_head)
        self.value_head = nn.Linear(d_model, 1) if self.has_value_head else None

    def hidden_states(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return causal Transformer representations before either output head."""
        x = self.token_embed(input_ids) + self.type_embed(token_type_ids)
        x = self.pos_encoder(x)
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=~attention_mask)
        return self.norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_values: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden_states(input_ids, token_type_ids, attention_mask)
        logits = self.lm_head(hidden)
        if not return_values:
            return logits
        if self.value_head is None:
            raise RuntimeError("return_values=True requires a model configured with value_head=True")
        return logits, self.value_head(hidden).squeeze(-1)


@dataclass
class EvalMetrics:
    loss: float
    action_accuracy: float
    token_loss: float = float("nan")
    value_loss: float = float("nan")
    value_mae: float = float("nan")
    wall_mask_accuracy: float = float("nan")
    visible_goal_accuracy: float = float("nan")
    goal_row_accuracy: float = float("nan")
    goal_column_accuracy: float = float("nan")
    complete_observation_accuracy: float = float("nan")
    reward_accuracy: float = float("nan")
    return_to_go_accuracy: float = float("nan")
    all_supervised_accuracy: float = float("nan")


def shifted_loss_and_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    with torch.no_grad():
        valid = shift_labels != ignore_index
        if valid.any():
            preds = shift_logits.argmax(dim=-1)
            acc = (preds[valid] == shift_labels[valid]).float().mean().item()
        else:
            acc = float("nan")
    return loss, acc


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    value_loss_weight: float = 1.0,
    progress_every: int = 0,
    phase: str = "epoch",
) -> EvalMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_token_loss = 0.0
    total_value_loss = 0.0
    value_absolute_error = 0.0
    value_count = 0
    steps = 0
    correct = {name: 0 for name in (
        "action", "wall", "visible", "row", "column", "reward", "return_to_go",
        "all", "complete",
    )}
    totals = {name: 0 for name in correct}

    started = time.perf_counter()
    batch_count = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        value_targets = batch["value_targets"].to(device)

        with torch.set_grad_enabled(training):
            if getattr(model, "has_value_head", False):
                logits, predicted_values = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_values=True,
                )
            else:
                logits = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
                predicted_values = None
            token_loss, acc = shifted_loss_and_accuracy(logits, labels)
            value_mask = torch.isfinite(value_targets) & attention_mask
            if predicted_values is not None and value_mask.any():
                value_loss = F.smooth_l1_loss(
                    predicted_values[value_mask], value_targets[value_mask]
                )
                loss = token_loss + value_loss_weight * value_loss
            else:
                value_loss = token_loss.new_zeros(())
                loss = token_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += float(loss.item())
        total_token_loss += float(token_loss.item())
        total_value_loss += float(value_loss.item())
        with torch.no_grad():
            if predicted_values is not None and value_mask.any():
                value_absolute_error += float(
                    torch.abs(predicted_values[value_mask] - value_targets[value_mask]).sum().item()
                )
                value_count += int(value_mask.sum().item())
            predictions = logits[:, :-1].argmax(dim=-1)
            targets = labels[:, 1:]
            target_types = token_type_ids[:, 1:]
            valid = targets != -100
            matches = predictions == targets
            masks = {
                "action": valid & (target_types == 5),
                "wall": valid & (target_types == 2),
                "visible": valid & (target_types == 3),
                "reward": valid & (target_types == 6),
                "return_to_go": valid & (target_types == 7),
                "all": valid,
            }
            delta_positions = valid & (target_types == 4)
            previous_types = torch.full_like(target_types, -1)
            previous_types[:, 1:] = target_types[:, :-1]
            masks["row"] = delta_positions & (previous_types == 3)
            masks["column"] = delta_positions & (previous_types == 4)
            for name, metric_mask in masks.items():
                correct[name] += int((matches & metric_mask).sum().item())
                totals[name] += int(metric_mask.sum().item())
            complete_mask = masks["wall"][:, :-3] & masks["visible"][:, 1:-2] & masks["row"][:, 2:-1] & masks["column"][:, 3:]
            complete_match = (matches[:, :-3] & matches[:, 1:-2] & matches[:, 2:-1] & matches[:, 3:])
            correct["complete"] += int((complete_mask & complete_match).sum().item())
            totals["complete"] += int(complete_mask.sum().item())
        steps += 1
        if progress_every > 0 and (batch_index % progress_every == 0 or batch_index == batch_count):
            elapsed = time.perf_counter() - started
            rate = batch_index / max(elapsed, 1e-9)
            remaining = (batch_count - batch_index) / max(rate, 1e-9)
            print(
                f"  {phase}: {batch_index}/{batch_count} batches "
                f"({100.0 * batch_index / max(batch_count, 1):5.1f}%) | "
                f"elapsed {elapsed / 60:.1f} min | ETA {remaining / 60:.1f} min",
                flush=True,
            )

    ratio = lambda name: correct[name] / totals[name] if totals[name] else float("nan")
    return EvalMetrics(loss=total_loss / max(steps, 1), action_accuracy=ratio("action"),
                       token_loss=total_token_loss / max(steps, 1),
                       value_loss=(total_value_loss / max(steps, 1)
                                   if getattr(model, "has_value_head", False) else float("nan")),
                       value_mae=(value_absolute_error / value_count if value_count else float("nan")),
                       wall_mask_accuracy=ratio("wall"), visible_goal_accuracy=ratio("visible"),
                       goal_row_accuracy=ratio("row"), goal_column_accuracy=ratio("column"),
                       complete_observation_accuracy=ratio("complete"),
                       reward_accuracy=ratio("reward"),
                       return_to_go_accuracy=ratio("return_to_go"),
                       all_supervised_accuracy=ratio("all"))


def decode_token(token_id: int, action_token_offset: int, action_token_names: list[str]) -> str:
    if token_id < action_token_offset:
        return f"STATE_{token_id}"
    return f"ACTION_{action_token_names[token_id - action_token_offset]}"


def greedy_generate_actions(
    model: TinyCausalTransformer,
    dataset: MazeTokenDataset,
    device: torch.device,
    episode_index: int,
    max_steps: int = 16,
) -> dict[str, object]:
    input_ids_full = dataset.input_ids[episode_index].numpy()
    labels_full = dataset.labels[episode_index].numpy()
    token_types_full = dataset.token_type_ids[episode_index].numpy()
    seq_len = int(dataset.sequence_lengths[episode_index].item())

    action_positions = np.flatnonzero(
        (labels_full[:seq_len] != -100) & (token_types_full[:seq_len] == 5)
    )
    prompt_end = int(action_positions[0]) if len(action_positions) else seq_len
    generated = input_ids_full[:prompt_end].tolist()
    generated_types = token_types_full[:prompt_end].tolist()
    target_actions = []
    pred_actions = []

    model.eval()
    for step_idx, action_pos in enumerate(action_positions[:max_steps]):
        x = torch.tensor([generated], dtype=torch.long, device=device)
        t = torch.tensor([generated_types], dtype=torch.long, device=device)
        m = torch.ones_like(x, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = model(input_ids=x, token_type_ids=t, attention_mask=m)
        action_logits = logits[0, -1, dataset.action_token_offset : dataset.action_token_offset + len(dataset.action_token_names)]
        next_token = int(dataset.action_token_offset + action_logits.argmax().item())
        target_token = int(labels_full[action_pos])
        pred_actions.append(next_token)
        target_actions.append(target_token)

        # Append predicted action, but keep the true next state as context so rollout stays on-manifold.
        generated.append(next_token)
        generated_types.append(5)
        next_state_pos = int(action_pos + 1)
        if next_state_pos >= seq_len:
            break
        generated.append(int(input_ids_full[next_state_pos]))
        generated_types.append(int(token_types_full[next_state_pos]))

    return {
        "episode_id": int(dataset.episode_ids[episode_index].item()),
        "target_action_tokens": target_actions,
        "pred_action_tokens": pred_actions,
        "target_actions_named": [
            decode_token(tok, dataset.action_token_offset, dataset.action_token_names) for tok in target_actions
        ],
        "pred_actions_named": [
            decode_token(tok, dataset.action_token_offset, dataset.action_token_names) for tok in pred_actions
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny causal transformer on discrete Maze2D tokens.")
    parser.add_argument("--dataset", type=Path, default=Path("maze2d_discrete_8x8_transformer_ready.npz"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-episodes", type=int, default=4096)
    parser.add_argument("--max-val-episodes", type=int, default=512)
    parser.add_argument("--use-predefined-split", action="store_true")
    parser.add_argument("--train-mask-key", type=str, default="train_mask")
    parser.add_argument("--val-mask-key", type=str, default="eval_mask")
    parser.add_argument("--cv-fold", type=int, default=None)
    parser.add_argument("--fold-ids-key", type=str, default="fold_ids")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--value-head", action="store_true")
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--progress-every", type=int, default=20,
        help="Print batch progress every N batches; use 0 to disable.",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/maze2d_discrete_transformer_tiny.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.dataset, allow_pickle=True)
    num_episodes = int(data["input_ids"].shape[0])

    if args.cv_fold is not None:
        if args.fold_ids_key not in data.files:
            raise KeyError(f"Requested cv fold split, but dataset is missing {args.fold_ids_key!r}.")
        fold_ids = np.asarray(data[args.fold_ids_key], dtype=np.int32)
        train_idx_all = np.flatnonzero(fold_ids != int(args.cv_fold))
        val_idx_all = np.flatnonzero(fold_ids == int(args.cv_fold))
        train_count = min(args.max_train_episodes, int(train_idx_all.size))
        val_count = min(args.max_val_episodes, int(val_idx_all.size))
        train_idx = train_idx_all[:train_count]
        val_idx = val_idx_all[:val_count]
    elif args.use_predefined_split:
        if args.train_mask_key not in data.files or args.val_mask_key not in data.files:
            raise KeyError(
                f"Requested predefined split, but dataset is missing {args.train_mask_key!r} "
                f"or {args.val_mask_key!r}."
            )
        train_idx_all = np.flatnonzero(np.asarray(data[args.train_mask_key], dtype=np.bool_))
        val_idx_all = np.flatnonzero(np.asarray(data[args.val_mask_key], dtype=np.bool_))
        train_count = min(args.max_train_episodes, int(train_idx_all.size))
        val_count = min(args.max_val_episodes, int(val_idx_all.size))
        train_idx = train_idx_all[:train_count]
        val_idx = val_idx_all[:val_count]
    else:
        perm = np.random.RandomState(args.seed).permutation(num_episodes)
        val_count = min(args.max_val_episodes, max(1, num_episodes // 10))
        train_count = min(args.max_train_episodes, num_episodes - val_count)
        train_idx = perm[:train_count]
        val_idx = perm[train_count : train_count + val_count]

    train_ds = MazeTokenDataset(args.dataset, train_idx)
    val_ds = MazeTokenDataset(args.dataset, val_idx)
    if args.value_head and not torch.isfinite(train_ds.value_targets).any():
        raise ValueError("--value-head requires a dataset containing finite value_targets")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TinyCausalTransformer(
        vocab_size=train_ds.vocab_size,
        max_seq_len=train_ds.input_ids.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        num_token_types=train_ds.num_token_types,
        value_head=args.value_head,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "train_episodes": len(train_ds),
                "val_episodes": len(val_ds),
                "seq_len": int(train_ds.input_ids.shape[1]),
                "vocab_size": train_ds.vocab_size,
                "device": str(device),
                "value_head": args.value_head,
                "value_loss_weight": args.value_loss_weight,
                "cpu_threads": torch.get_num_threads(),
            },
            indent=2,
        )
    )

    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, optimizer, device,
            value_loss_weight=args.value_loss_weight,
            progress_every=args.progress_every,
            phase=f"epoch {epoch}/{args.epochs} train",
        )
        val_metrics = run_epoch(
            model, val_loader, None, device,
            value_loss_weight=args.value_loss_weight,
            progress_every=args.progress_every,
            phase=f"epoch {epoch}/{args.epochs} validation",
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics.loss,
            "train_action_accuracy": train_metrics.action_accuracy,
            "val_loss": val_metrics.loss,
            "val_action_accuracy": val_metrics.action_accuracy,
            "train_token_loss": train_metrics.token_loss,
            "val_token_loss": val_metrics.token_loss,
            "train_value_loss": train_metrics.value_loss,
            "val_value_loss": val_metrics.value_loss,
            "train_value_mae": train_metrics.value_mae,
            "val_value_mae": val_metrics.value_mae,
        }
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for name in ("wall_mask_accuracy", "visible_goal_accuracy", "goal_row_accuracy",
                         "goal_column_accuracy", "complete_observation_accuracy", "reward_accuracy",
                         "return_to_go_accuracy", "all_supervised_accuracy"):
                row[f"{prefix}_{name}"] = getattr(metrics, name)
        history.append(row)
        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.action_accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.action_accuracy:.4f} | "
            f"val_value_mae={val_metrics.value_mae:.4f} | "
            f"epoch_time={(time.perf_counter() - epoch_started) / 60:.1f} min",
            flush=True,
        )
        if val_metrics.loss < best_val:
            best_val = val_metrics.loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "vocab_size": train_ds.vocab_size,
                        "max_seq_len": int(train_ds.input_ids.shape[1]),
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_layers": args.num_layers,
                        "dim_feedforward": args.ffn_dim,
                        "dropout": args.dropout,
                        "num_token_types": train_ds.num_token_types,
                        "value_head": args.value_head,
                    },
                    "history": history,
                    "dataset_metadata": {
                        key: data[key].tolist() for key in (
                            "dataset_version", "supervision", "observation_encoding", "observation_group_size",
                            "wall_token_offset", "wall_token_count", "visible_goal_token_offset",
                             "visible_goal_token_count", "goal_delta_token_offset", "goal_delta_min",
                            "goal_delta_max", "action_token_offset", "bos_token_id",
                            "reward_token_offset", "reward_token_count", "reward_values_cents",
                            "return_to_go_token_offset", "return_to_go_token_count",
                            "return_to_go_min_cents", "return_to_go_max_cents", "reward_scale",
                            "discount_factor", "token_type_reward", "token_type_return_to_go",
                            "value_target_name", "value_target_units",
                        ) if key in data.files
                    },
                },
                args.output,
            )

    preview = greedy_generate_actions(model, val_ds, device, episode_index=0)
    print("preview rollout:", json.dumps(preview, indent=2))
    print("saved checkpoint:", args.output)


if __name__ == "__main__":
    main()
