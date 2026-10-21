"""Rerun only the fully observed CV evaluation and its result-display cells."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "Full_dataset_pipeline.ipynb"

# Imports and definitions are needed in a fresh kernel, but their stored outputs
# are not changed. In particular, cell 20 (training) and cells 42+ (partial
# observation) are never executed.
DEPENDENCY_CELLS = [1, 12, 13, 14, 15, 16, 17, 19, 21, 22]
RESULT_CELLS = [23, 25, 26, 27, 28]
SELECTED_CELLS = DEPENDENCY_CELLS + RESULT_CELLS


def main() -> None:
    original = nbformat.read(NOTEBOOK_PATH, as_version=4)
    original_sources = {i: original.cells[i].source for i in SELECTED_CELLS}
    selected = nbformat.v4.new_notebook(
        cells=[copy.deepcopy(original.cells[i]) for i in SELECTED_CELLS],
        metadata=copy.deepcopy(original.metadata),
    )
    client = NotebookClient(
        selected,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_DIR)}},
    )

    with client.setup_kernel():
        for selected_index, original_index in enumerate(SELECTED_CELLS):
            print(f"Running notebook cell {original_index}...", flush=True)
            client.execute_cell(selected.cells[selected_index], selected_index)
            print(f"Finished notebook cell {original_index}.", flush=True)

    # Read raw JSON so cells without IDs and all unrelated cell metadata remain
    # untouched. Abort if an executed cell's source changed in the meantime.
    updated = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for index, source in original_sources.items():
        if "".join(updated["cells"][index]["source"]) != source:
            raise RuntimeError(f"Notebook cell {index} changed during the run; outputs were not saved.")

    for original_index in RESULT_CELLS:
        selected_index = SELECTED_CELLS.index(original_index)
        updated["cells"][original_index]["outputs"] = selected.cells[selected_index].outputs
        updated["cells"][original_index]["execution_count"] = selected.cells[selected_index].execution_count

    NOTEBOOK_PATH.write_text(json.dumps(updated, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated outputs only for notebook cells {RESULT_CELLS}.", flush=True)


if __name__ == "__main__":
    main()
