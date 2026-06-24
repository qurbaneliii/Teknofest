from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASETS = {"MASTER": "YARISMA_TRAIN_MASTER.csv", "KANSER": "YARISMA_TRAIN_KANSER.csv", "CFTR": "YARISMA_TRAIN_CFTR.csv", "PAH": "YARISMA_TRAIN_PAH.csv"}


def load_labeled_competition_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    result: dict[str, pd.DataFrame] = {}
    for name, filename in DATASETS.items():
        candidates = list(root.rglob(filename))
        if not candidates:
            raise FileNotFoundError(f"Missing {filename} under {root}")
        frame = pd.read_csv(candidates[0])
        if "Label" not in frame:
            raise ValueError(f"{candidates[0]} has no Label column and cannot be used for internal evaluation.")
        result[name] = frame
    return result


def find_unlabeled_test_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    found = []
    for path in root.rglob("*.csv"):
        relative_parts = path.relative_to(root).parts
        if any(part in {"artifacts", "reports", "outputs", ".git"} for part in relative_parts):
            continue
        if "test" not in path.name.lower():
            continue
        try:
            columns = pd.read_csv(path, nrows=1).columns
        except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
            continue
        if "Label" not in columns:
            found.append(path)
    return found
