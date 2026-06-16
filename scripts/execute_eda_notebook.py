from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


INPUT_NOTEBOOK = Path("reports/eda/TEKNOFEST2026_EDA.ipynb")
OUTPUT_NOTEBOOK = Path("reports/eda/TEKNOFEST2026_EDA_EXECUTED.ipynb")


def main() -> None:
    nb = nbformat.read(INPUT_NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    client.execute()
    nbformat.write(nb, OUTPUT_NOTEBOOK)
    print(f"Executed notebook written to: {OUTPUT_NOTEBOOK.resolve()}")


if __name__ == "__main__":
    main()
