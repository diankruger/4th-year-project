from __future__ import annotations

import json
from pathlib import Path

import numpy as np


WALL = 10
EMPTY = 11
GOAL = 12


# These are the built-in Maze2D layouts from D4RL's pointmaze definitions.
# Kept here as plain strings so the extractor stays simple and easy to read.
MAZE_SPECS = {
    "open": [
        "#######",
        "#OOOOO#",
        "#OOGOO#",
        "#OOOOO#",
        "#######",
    ],
    "u_maze": [
        "#####",
        "#GOO#",
        "###O#",
        "#OOO#",
        "#####",
    ],
    "u_maze_eval": [
        "#####",
        "#OOG#",
        "#O###",
        "#OOO#",
        "#####",
    ],
    "small": [
        "######",
        "#OOOO#",
        "#O##O#",
        "#OOOO#",
        "######",
    ],
    "medium": [
        "########",
        "#OO##OO#",
        "#OO#OOO#",
        "##OOO###",
        "#OO#OOO#",
        "#O#OO#O#",
        "#OOO#OG#",
        "########",
    ],
    "medium_eval": [
        "########",
        "#OOOOOG#",
        "#O#O##O#",
        "#OOOO#O#",
        "###OO###",
        "#OOOOOO#",
        "#OO##OO#",
        "########",
    ],
    "large": [
        "############",
        "#OOOO#OOOOO#",
        "#O##O#O#O#O#",
        "#OOOOOO#OOO#",
        "#O####O###O#",
        "#OO#O#OOOOO#",
        "##O#O#O#O###",
        "#OO#OOO#OGO#",
        "############",
    ],
    "large_eval": [
        "############",
        "#OO#OOO#OGO#",
        "##O###O#O#O#",
        "#OO#O#OOOOO#",
        "#O##O#OO##O#",
        "#OOOOOO#OOO#",
        "#O##O#O#O###",
        "#OOOO#OOOOO#",
        "############",
    ],
}


def to_numeric_grid(rows: list[str]) -> np.ndarray:
    """Convert a human-readable maze into the integer grid used by Maze2D."""
    grid = np.zeros((len(rows), len(rows[0])), dtype=np.int32)
    value_map = {"#": WALL, "O": EMPTY, "0": EMPTY, " ": EMPTY, "G": GOAL}

    for r, row in enumerate(rows):
        for c, char in enumerate(row):
            if char not in value_map:
                raise ValueError(f"Unknown maze character {char!r} at row {r}, col {c}")
            grid[r, c] = value_map[char]

    return grid


def list_cells(grid: np.ndarray, value: int) -> list[list[int]]:
    """Return [row, col] pairs for all cells with a given value."""
    return [[int(r), int(c)] for r, c in np.argwhere(grid == value)]


def build_graph(grid: np.ndarray) -> dict[str, object]:
    """Create a simple 4-neighbor graph over all traversable cells."""
    traversable = np.argwhere(grid != WALL)
    node_lookup = {tuple(cell): idx for idx, cell in enumerate(map(tuple, traversable.tolist()))}

    nodes = []
    edges = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for node_id, (row, col) in enumerate(map(tuple, traversable.tolist())):
        nodes.append(
            {
                "id": node_id,
                "row": int(row),
                "col": int(col),
                "kind": "goal" if grid[row, col] == GOAL else "free",
            }
        )

        for dr, dc in directions:
            neighbor = (row + dr, col + dc)
            if neighbor in node_lookup:
                edges.append([node_id, node_lookup[neighbor]])

    return {"nodes": nodes, "edges": edges}


def build_record(name: str, rows: list[str]) -> dict[str, object]:
    """Build one maze record in a planning-friendly format."""
    grid = to_numeric_grid(rows)
    occupancy = (grid == WALL).astype(np.int8)
    graph = build_graph(grid)

    return {
        "name": name,
        "ascii": rows,
        "height": int(grid.shape[0]),
        "width": int(grid.shape[1]),
        "grid_values": {"wall": WALL, "empty": EMPTY, "goal": GOAL},
        "grid": grid.tolist(),
        "occupancy_grid": occupancy.tolist(),
        "free_cells": list_cells(grid, EMPTY),
        "goal_cells": list_cells(grid, GOAL),
        "wall_cells": list_cells(grid, WALL),
        "graph": graph,
    }


def write_json(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.write_text(json.dumps({"mazes": records}, indent=2), encoding="utf-8")


def write_npz(records: list[dict[str, object]], output_path: Path) -> None:
    arrays = {}
    for record in records:
        name = record["name"]
        arrays[f"{name}_grid"] = np.asarray(record["grid"], dtype=np.int32)
        arrays[f"{name}_occupancy"] = np.asarray(record["occupancy_grid"], dtype=np.int8)
    np.savez(output_path, **arrays)


def write_summary(records: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Maze2D Layout Dataset",
        "",
        "This folder contains the built-in Maze2D layouts extracted into clean planning-friendly formats.",
        "",
        "Files:",
        "- `maze2d_layouts.json`: full metadata, grids, cell lists, and adjacency graphs",
        "- `maze2d_layouts.npz`: compact NumPy arrays for the integer grids and occupancy masks",
        "",
        "Grid encoding:",
        f"- wall = {WALL}",
        f"- empty = {EMPTY}",
        f"- goal = {GOAL}",
        "",
        "Mazes:",
    ]

    for record in records:
        lines.append(
            f"- {record['name']}: {record['height']}x{record['width']}, "
            f"{len(record['free_cells'])} free cells, {len(record['goal_cells'])} goal cells"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = Path("maze_layouts")
    output_dir.mkdir(exist_ok=True)

    records = [build_record(name, rows) for name, rows in MAZE_SPECS.items()]

    write_json(records, output_dir / "maze2d_layouts.json")
    write_npz(records, output_dir / "maze2d_layouts.npz")
    write_summary(records, output_dir / "README.md")

    print(f"Wrote {len(records)} maze layouts to {output_dir.resolve()}")
    for record in records:
        print(
            f"- {record['name']}: {record['height']}x{record['width']}, "
            f"free={len(record['free_cells'])}, goal={len(record['goal_cells'])}"
        )


if __name__ == "__main__":
    main()
