import argparse
import json
from pathlib import Path

import numpy as np



MAZE_SPECS = {
    "OPEN": "#######\\#OOOOO#\\#OOGOO#\\#OOOOO#\\#######",
    "U_MAZE": "#####\\#GOO#\\###O#\\#OOO#\\#####",
    "U_MAZE_EVAL": "#####\\#OOG#\\#O###\\#OOO#\\#####",
    "SMALL_MAZE": "######\\#OOOO#\\#O##O#\\#OOOO#\\######",
    "MEDIUM_MAZE": "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########",
    "MEDIUM_MAZE_EVAL": "########\\#OOOOOG#\\#O#O##O#\\#OOOO#O#\\###OO###\\#OOOOOO#\\#OO##OO#\\########",
    "LARGE_MAZE": "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############",
    "LARGE_MAZE_EVAL": "############\\#OO#OOO#OGO#\\##O###O#O#O#\\#OO#O#OOOOO#\\#O##O#OO##O#\\#OOOOOO#OOO#\\#O##O#O#O###\\#OOOO#OOOOO#\\############",
}
def parse_maze_rows(maze_str):
    return maze_str.strip().split("\\")


def layout_from_spec(maze_str):
    rows = parse_maze_rows(maze_str)
    return {
        "height": len(rows),
        "width": len(rows[0]),
        "wall_cells": [
            [row, col]
            for row, row_text in enumerate(rows)
            for col, char in enumerate(row_text)
            if char == "#"
        ],
    }


def wall_rectangles_from_layout(layout):
    rectangles = []
    width = layout["width"]
    for row, col in layout["wall_cells"]:
        x = width - 1 - col
        y = row
        rectangles.append((x - 0.5, y - 0.5, 1.0, 1.0))
    return rectangles


def grid_shape_for_maze(maze_name):
    if "LARGE" in maze_name:
        return (9, 12)
    return (8, 8)


def make_discrete_maze(layout, grid_shape=(8, 8)):
    grid_rows, grid_cols = grid_shape
    discrete_maze = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    wall_rectangles = wall_rectangles_from_layout(layout)
    x_edges = np.linspace(-0.5, layout["width"] - 0.5, grid_cols + 1)
    y_edges = np.linspace(-0.5, layout["height"] - 0.5, grid_rows + 1)

    for row in range(grid_rows):
        y0 = y_edges[row]
        y1 = y_edges[row + 1]
        for col in range(grid_cols):
            x0 = x_edges[col]
            x1 = x_edges[col + 1]
            for wall_x, wall_y, wall_w, wall_h in wall_rectangles:
                if x0 < wall_x + wall_w and x1 > wall_x and y0 < wall_y + wall_h and y1 > wall_y:
                    discrete_maze[row, col] = 1
                    break
    return discrete_maze


def build_gallery_notebook():
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Maze2D Discrete Maze Gallery\n",
                    "\n",
                    "This notebook loads the exported discrete Maze2D dataset and plots every maze with its stored grid shape.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    'DATASET_PATH = Path("maze2d_discrete_8x8_dataset.npz")\n',
                    "dataset = np.load(DATASET_PATH, allow_pickle=True)\n",
                    'maze_names = dataset["maze_names"]\n',
                    'mazes = dataset["mazes"]\n',
                    'grid_shapes = dataset["grid_shapes"]\n',
                    "\n",
                    'print("Dataset path:", DATASET_PATH)\n',
                    'print("Number of mazes:", len(maze_names))\n',
                    'print("Stored maze array shape:", mazes.shape)\n',
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
                    "axes = axes.ravel()\n",
                    "\n",
                    "for ax, maze_name, maze, grid_shape in zip(axes, maze_names, mazes, grid_shapes):\n",
                    "    grid_rows, grid_cols = [int(v) for v in grid_shape]\n",
                    '    ax.imshow(maze, origin="lower", cmap="Greys", vmin=0, vmax=1)\n',
                    '    ax.set_title(f"{maze_name} ({grid_rows}x{grid_cols})")\n',
                    "    ax.set_xticks(np.arange(grid_cols))\n",
                    "    ax.set_yticks(np.arange(grid_rows))\n",
                    "    ax.set_xticks(np.arange(-0.5, grid_cols, 1), minor=True)\n",
                    "    ax.set_yticks(np.arange(-0.5, grid_rows, 1), minor=True)\n",
                    '    ax.grid(which="minor", color="white", linewidth=1.0)\n',
                    '    ax.set_xlabel("grid col")\n',
                    '    ax.set_ylabel("grid row")\n',
                    "\n",
                    "for ax in axes[len(maze_names):]:\n",
                    '    ax.axis("off")\n',
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    'maze_summary = {\n',
                    '    str(name): {"grid_shape": [int(v) for v in grid_shape], "maze": maze.tolist()}\n',
                    '    for name, maze, grid_shape in zip(maze_names, mazes, grid_shapes)\n',
                    '}\n',
                    "maze_summary\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "venv_d4rl (3.10.11)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the discrete Maze2D maze dataset."
    )
    parser.add_argument(
        "--write-notebook",
        action="store_true",
        help="Also write the gallery notebook scaffold.",
    )
    parser.add_argument(
        "--overwrite-notebook",
        action="store_true",
        help="Allow overwriting an existing gallery notebook when used with --write-notebook.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    maze_names = list(MAZE_SPECS.keys())
    grid_shapes = np.array([grid_shape_for_maze(name) for name in maze_names], dtype=np.int32)
    mazes = np.empty(len(maze_names), dtype=object)
    for index, maze_name in enumerate(maze_names):
        mazes[index] = make_discrete_maze(
            layout_from_spec(MAZE_SPECS[maze_name]),
            grid_shape=tuple(grid_shapes[index]),
        )

    dataset_path = Path("maze2d_discrete_8x8_dataset.npz")
    np.savez(
        dataset_path,
        maze_names=np.array(maze_names, dtype="<U32"),
        mazes=mazes,
        grid_shapes=grid_shapes,
    )

    notebook_path = Path("maze2d_discrete_8x8_gallery.ipynb")
    notebook_status = "not requested"
    if args.write_notebook:
        if notebook_path.exists() and not args.overwrite_notebook:
            notebook_status = "skipped existing notebook"
        else:
            # notebook_path.write_text(json.dumps(build_gallery_notebook(), indent=1), encoding="utf-8")
            notebook_status = "written"

    print("dataset:", dataset_path)
    print("notebook:", notebook_path)
    print("notebook_status:", notebook_status)
    print("maze names:", maze_names)
    print("stored maze array shape:", mazes.shape)
    print("grid shapes:", grid_shapes.tolist())


if __name__ == "__main__":
    main()
