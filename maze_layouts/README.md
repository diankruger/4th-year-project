# Maze2D Layout Dataset

This folder contains the built-in Maze2D layouts extracted into clean planning-friendly formats.

Files:
- `maze2d_layouts.json`: full metadata, grids, cell lists, and adjacency graphs
- `maze2d_layouts.npz`: compact NumPy arrays for the integer grids and occupancy masks

Grid encoding:
- wall = 10
- empty = 11
- goal = 12

Mazes:
- open: 5x7, 14 free cells, 1 goal cells
- u_maze: 5x5, 6 free cells, 1 goal cells
- u_maze_eval: 5x5, 6 free cells, 1 goal cells
- small: 5x6, 10 free cells, 0 goal cells
- medium: 8x8, 25 free cells, 1 goal cells
- medium_eval: 8x8, 25 free cells, 1 goal cells
- large: 9x12, 45 free cells, 1 goal cells
- large_eval: 9x12, 46 free cells, 1 goal cells
