from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import TABLES_DIR
from teknofest.data_prep import load_datasets, overlap_summary, prepare_data


REQUIRED_GROUP_PREFIXES = ("AL_", "CAT_", "EK_")
REQUIRED_COLUMNS = {"Variant_ID", "Label", "AA_1", "AA_2"}


def discover_data_dir(root: str | Path = ".") -> Path:
    root = Path(root)
    matches = list(root.rglob("YARISMA_TRAIN_MASTER.csv"))
    if not matches:
        raise FileNotFoundError("Could not find YARISMA_TRAIN_MASTER.csv under repository root.")
    return matches[0].parent


def validate_schema(datasets: dict[str, pd.DataFrame]) -> None:
    master_cols = list(datasets["master"].columns)
    for name, df in datasets.items():
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")
        for prefix in REQUIRED_GROUP_PREFIXES:
            if not any(c.startswith(prefix) for c in df.columns):
                raise ValueError(f"{name} has no {prefix} columns")
        if list(df.columns) != master_cols:
            raise ValueError(f"{name} schema differs from MASTER")


def write_data_diagnostics(data_dir: str | Path, out_dir: str | Path = TABLES_DIR) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(data_dir)
    validate_schema(datasets)
    prepared = prepare_data(data_dir)

    rows = []
    miss_rows = []
    for name, df in datasets.items():
        rows.append(
            {
                "dataset": name.upper(),
                "rows": len(df),
                "columns": len(df.columns),
                "duplicate_variant_id": int(df["Variant_ID"].duplicated().sum()),
                "pathogenic": int(df["Label"].sum()),
                "benign": int((df["Label"] == 0).sum()),
                "pathogenic_rate": float(df["Label"].mean()),
            }
        )
        missing = df.isna().mean().sort_values(ascending=False)
        for col, rate in missing.items():
            miss_rows.append({"dataset": name.upper(), "column": col, "missing_rate": float(rate)})

    pd.DataFrame(rows).to_csv(out / "dataset_summary.csv", index=False)
    pd.DataFrame(miss_rows).to_csv(out / "missingness_summary.csv", index=False)
    overlap_summary(prepared).to_csv(out / "overlap_summary.csv", index=False)

