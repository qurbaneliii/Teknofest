from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_FILES = {
    "master": "YARISMA_TRAIN_MASTER.csv",
    "kanser": "YARISMA_TRAIN_KANSER.csv",
    "pah": "YARISMA_TRAIN_PAH.csv",
    "cftr": "YARISMA_TRAIN_CFTR.csv",
}

SEED = 42


@dataclass(frozen=True)
class PreparedData:
    master: pd.DataFrame
    kanser: pd.DataFrame
    pah: pd.DataFrame
    cftr: pd.DataFrame
    master_only_mask: pd.Series
    master_shared_mask: pd.Series
    kanser_unique: pd.DataFrame
    pah_unique: pd.DataFrame
    cftr_unique: pd.DataFrame
    al_cols: list[str]
    al_raw: list[str]


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)


def load_datasets(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_path = Path(data_dir)
    missing = [
        filename
        for filename in DATASET_FILES.values()
        if not (data_path / filename).exists()
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required dataset file(s) in {data_path}: {joined}")

    return {
        name: pd.read_csv(data_path / filename)
        for name, filename in DATASET_FILES.items()
    }


def prepare_data(data_dir: str | Path) -> PreparedData:
    set_seeds()
    datasets = load_datasets(data_dir)

    master = datasets["master"]
    kanser = datasets["kanser"]
    pah = datasets["pah"]
    cftr = datasets["cftr"]

    master_ids = set(master.Variant_ID)
    kanser_ids = set(kanser.Variant_ID)
    pah_ids = set(pah.Variant_ID)
    cftr_ids = set(cftr.Variant_ID)
    panel_ids = kanser_ids | pah_ids | cftr_ids

    master_only_mask = master.Variant_ID.isin(master_ids - panel_ids)
    master_shared_mask = master.Variant_ID.isin(master_ids & panel_ids)
    kanser_unique_mask = ~kanser.Variant_ID.isin(master_ids)
    pah_unique_mask = ~pah.Variant_ID.isin(master_ids)
    cftr_unique_mask = ~cftr.Variant_ID.isin(master_ids)

    for df in (master, kanser, pah, cftr):
        df.drop(columns=["AL_185"], inplace=True)

    kanser_unique = kanser[kanser_unique_mask].copy()
    pah_unique = pah[pah_unique_mask].copy()
    cftr_unique = cftr[cftr_unique_mask].copy()

    al_cols = [c for c in master.columns if c.startswith("AL_")]
    al_raw = [f"AL_{i}" for i in range(1, 27)]

    return PreparedData(
        master=master,
        kanser=kanser,
        pah=pah,
        cftr=cftr,
        master_only_mask=master_only_mask,
        master_shared_mask=master_shared_mask,
        kanser_unique=kanser_unique,
        pah_unique=pah_unique,
        cftr_unique=cftr_unique,
        al_cols=al_cols,
        al_raw=al_raw,
    )


def pathogenic_rate(df: pd.DataFrame) -> float | None:
    if "Label" not in df.columns or df.empty:
        return None
    return float(df["Label"].mean())


def overlap_summary(prepared: PreparedData) -> pd.DataFrame:
    master_ids = set(prepared.master.Variant_ID)
    kanser_ids = set(prepared.kanser.Variant_ID)
    pah_ids = set(prepared.pah.Variant_ID)
    cftr_ids = set(prepared.cftr.Variant_ID)

    rows = [
        ("MASTER", len(prepared.master), pathogenic_rate(prepared.master)),
        ("KANSER", len(prepared.kanser), pathogenic_rate(prepared.kanser)),
        ("PAH", len(prepared.pah), pathogenic_rate(prepared.pah)),
        ("CFTR", len(prepared.cftr), pathogenic_rate(prepared.cftr)),
        ("MASTER_intersect_KANSER", len(master_ids & kanser_ids), None),
        ("MASTER_intersect_PAH", len(master_ids & pah_ids), None),
        ("MASTER_intersect_CFTR", len(master_ids & cftr_ids), None),
        ("KANSER_intersect_PAH", len(kanser_ids & pah_ids), None),
        ("KANSER_intersect_CFTR", len(kanser_ids & cftr_ids), None),
        ("PAH_intersect_CFTR", len(pah_ids & cftr_ids), None),
        (
            "MASTER_only",
            int(prepared.master_only_mask.sum()),
            pathogenic_rate(prepared.master[prepared.master_only_mask]),
        ),
        (
            "MASTER_shared_with_panels",
            int(prepared.master_shared_mask.sum()),
            pathogenic_rate(prepared.master[prepared.master_shared_mask]),
        ),
        ("KANSER_unique", len(prepared.kanser_unique), pathogenic_rate(prepared.kanser_unique)),
        ("PAH_unique", len(prepared.pah_unique), pathogenic_rate(prepared.pah_unique)),
        ("CFTR_unique", len(prepared.cftr_unique), pathogenic_rate(prepared.cftr_unique)),
    ]
    return pd.DataFrame(rows, columns=["item", "n", "pathogenic_rate"])


def validate_first_section(prepared: PreparedData) -> dict[str, object]:
    return {
        "al_185_dropped": all(
            "AL_185" not in df.columns
            for df in (prepared.master, prepared.kanser, prepared.pah, prepared.cftr)
        ),
        "al_186_kept": "AL_186" in prepared.master.columns,
        "n_al_cols_after_drop": len(prepared.al_cols),
        "al_raw": prepared.al_raw,
    }
