import json
from pathlib import Path


NOTEBOOK_PATH = Path("maze2d_discrete_8x8_gallery.ipynb")


def make_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def make_code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    marker = "## Tiny transformer proof of concept"
    if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
        print("transformer visualisation section already present")
        return

    cells = [
        make_markdown_cell(
            "## Tiny transformer proof of concept\n\n"
            "This section loads a small causal transformer trained on the transformer-ready Maze2D token dataset and visualises what it learned.\n"
        ),
        make_code_cell(
            """import math
import torch
import torch.nn as nn

TRANSFORMER_CHECKPOINT_PATH = Path("checkpoints/maze2d_discrete_transformer_tiny_4ep.pt")


class NotebookPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NotebookTinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, num_token_types=6):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.type_embed = nn.Embedding(num_token_types, d_model)
        self.pos_encoder = NotebookPositionalEncoding(d_model=d_model, max_len=max_seq_len)
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

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.token_embed(input_ids) + self.type_embed(token_type_ids)
        x = self.pos_encoder(x)
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device), diagonal=1)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=~attention_mask)
        x = self.norm(x)
        return self.lm_head(x)


def shifted_action_metrics(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    if not torch.any(valid):
        return float("nan"), float("nan")
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    preds = shift_logits.argmax(dim=-1)
    acc = (preds[valid] == shift_labels[valid]).float().mean().item()
    return float(loss.item()), float(acc)


transformer_checkpoint = torch.load(TRANSFORMER_CHECKPOINT_PATH, map_location="cpu")
transformer_model = NotebookTinyCausalTransformer(**transformer_checkpoint["model_config"])
transformer_model.load_state_dict(transformer_checkpoint["model_state_dict"])
transformer_model.eval()
transformer_history = pd.DataFrame(transformer_checkpoint["history"])

transformer_tokens = np.load(TRANSFORMER_READY_DATASET_PATH, allow_pickle=True)
transformer_input_ids = torch.from_numpy(transformer_tokens["input_ids"].astype(np.int64))
transformer_labels = torch.from_numpy(transformer_tokens["labels"].astype(np.int64))
transformer_attention_mask = torch.from_numpy(transformer_tokens["attention_mask"].astype(np.bool_))
transformer_token_type_ids = torch.from_numpy(transformer_tokens["token_type_ids"].astype(np.int64))
transformer_sequence_lengths = transformer_tokens["sequence_lengths"].astype(np.int32)
transformer_episode_ids = transformer_tokens["episode_ids"].astype(np.int32)

with torch.no_grad():
    transformer_logits = transformer_model(
        input_ids=transformer_input_ids,
        token_type_ids=transformer_token_type_ids,
        attention_mask=transformer_attention_mask,
    )

full_loss, full_acc = shifted_action_metrics(transformer_logits, transformer_labels)
print({
    "checkpoint": str(TRANSFORMER_CHECKPOINT_PATH),
    "history_epochs": len(transformer_history),
    "dataset_action_loss": round(full_loss, 4),
    "dataset_action_accuracy": round(full_acc, 4),
})
"""
        ),
        make_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(transformer_history["epoch"], transformer_history["train_loss"], marker="o", label="train")
axes[0].plot(transformer_history["epoch"], transformer_history["val_loss"], marker="o", label="val")
axes[0].set_title("Tiny transformer loss")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("cross-entropy loss")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(transformer_history["epoch"], transformer_history["train_action_accuracy"], marker="o", label="train")
axes[1].plot(transformer_history["epoch"], transformer_history["val_action_accuracy"], marker="o", label="val")
axes[1].set_title("Tiny transformer action accuracy")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy")
axes[1].set_ylim(0.0, 1.0)
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

transformer_history
"""
        ),
        make_code_cell(
            """def state_token_to_cell(token_id):
    state_index = int(token_id) - STATE_TOKEN_OFFSET
    return np.array([state_index // trajectory_grid_shape[1], state_index % trajectory_grid_shape[1]], dtype=np.int16)


def greedy_rollout_from_start(start_cell, goal_cell, max_steps=16):
    generated_tokens = [MAZE_TOKEN_OFFSET, cell_to_state_token(start_cell, trajectory_grid_shape), cell_to_state_token(goal_cell, trajectory_grid_shape), cell_to_state_token(start_cell, trajectory_grid_shape)]
    generated_types = [TOKEN_TYPE_MAZE, TOKEN_TYPE_START, TOKEN_TYPE_GOAL, TOKEN_TYPE_STATE]
    path = [np.asarray(start_cell, dtype=np.int16)]

    for _ in range(max_steps):
        x = torch.tensor([generated_tokens], dtype=torch.long)
        t = torch.tensor([generated_types], dtype=torch.long)
        m = torch.ones_like(x, dtype=torch.bool)
        with torch.no_grad():
            logits = transformer_model(input_ids=x, token_type_ids=t, attention_mask=m)
        action_logits = logits[0, -1, ACTION_TOKEN_OFFSET:ACTION_TOKEN_OFFSET + len(ACTION_TOKEN_NAMES)].cpu().numpy()
        current_cell = path[-1]

        valid_action_indices = []
        for action_idx, delta in enumerate(ACTION_DELTAS):
            next_cell = current_cell + delta.astype(np.int16)
            inside = 0 <= next_cell[0] < trajectory_grid_shape[0] and 0 <= next_cell[1] < trajectory_grid_shape[1]
            free = inside and trajectory_maze[tuple(next_cell)] == 0
            if free:
                valid_action_indices.append(action_idx)

        best_valid_idx = max(valid_action_indices, key=lambda idx: float(action_logits[idx]))
        delta = ACTION_DELTAS[best_valid_idx].astype(np.int16)
        next_cell = current_cell + delta
        generated_tokens.append(int(ACTION_TOKEN_OFFSET + best_valid_idx))
        generated_types.append(TOKEN_TYPE_ACTION)
        generated_tokens.append(cell_to_state_token(next_cell, trajectory_grid_shape))
        generated_types.append(TOKEN_TYPE_STATE)
        path.append(next_cell)

        if np.array_equal(next_cell, goal_cell):
            break

    return np.asarray(path, dtype=np.int16)
"""
        ),
        make_code_cell(
            """rng = np.random.default_rng(7)
example_indices = rng.choice(len(trajectory_paths), size=6, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = np.asarray(axes).ravel()
summary_rows = []

for ax, example_idx in zip(axes, example_indices):
    true_path = np.asarray(trajectory_paths[int(example_idx)], dtype=np.int16)
    goal_cell = np.asarray(trajectory_goal_cells[int(example_idx)], dtype=np.int16)
    pred_path = greedy_rollout_from_start(true_path[0], goal_cell, max_steps=max(2 * len(true_path), 16))

    overlay = np.ones((trajectory_maze.shape[1], trajectory_maze.shape[0], 3), dtype=np.float32)
    overlay[trajectory_maze.T == 1] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ax.imshow(overlay, origin="lower", extent=(-0.5, trajectory_maze.shape[0] - 0.5, -0.5, trajectory_maze.shape[1] - 0.5))

    ax.plot(true_path[:, 0], true_path[:, 1], color="dodgerblue", linewidth=3.0, alpha=0.9, label="dataset path")
    ax.plot(pred_path[:, 0], pred_path[:, 1], color="crimson", linewidth=2.0, linestyle="--", alpha=0.9, label="model rollout")
    ax.scatter(true_path[0, 0], true_path[0, 1], c="limegreen", s=80, label="start")
    ax.scatter(goal_cell[0], goal_cell[1], c="gold", s=120, marker="X", edgecolors="black", linewidths=0.8, label="goal")
    ax.scatter(pred_path[-1, 0], pred_path[-1, 1], c="crimson", s=70, marker="s", label="model end")

    reached_goal = bool(np.array_equal(pred_path[-1], goal_cell))
    optimal_steps = int(max(len(true_path) - 1, 0))
    rollout_steps = int(max(len(pred_path) - 1, 0))
    ax.set_title(f"Ep {int(trajectory_episode_ids[int(example_idx)])} | goal={reached_goal} | {rollout_steps}/{optimal_steps} steps")
    ax.set_xticks(np.arange(trajectory_grid_shape[0]))
    ax.set_yticks(np.arange(trajectory_grid_shape[1]))
    ax.set_xticks(np.arange(-0.5, trajectory_grid_shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, trajectory_grid_shape[1], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.set_xlabel("maze_x / qpos[0]")
    ax.set_ylabel("maze_y / qpos[1]")

    summary_rows.append({
        "episode_id": int(trajectory_episode_ids[int(example_idx)]),
        "dataset_steps": optimal_steps,
        "rollout_steps": rollout_steps,
        "reached_goal": reached_goal,
        "final_cell_x": int(pred_path[-1, 0]),
        "final_cell_y": int(pred_path[-1, 1]),
    })

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

pd.DataFrame(summary_rows)
"""
        ),
    ]

    nb["cells"].extend(cells)
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("appended transformer visualisation section")


if __name__ == "__main__":
    main()
