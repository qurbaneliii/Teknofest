from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    train_files: List[str] = field(
        default_factory=lambda: [
            "YARISMA_TRAIN_CFTR.csv",
            "YARISMA_TRAIN_KANSER.csv",
            "YARISMA_TRAIN_MASTER.csv",
            "YARISMA_TRAIN_PAH.csv",
        ]
    )

    def full_paths(self) -> List[Path]:
        return [self.root / name for name in self.train_files]


@dataclass(frozen=True)
class FeatureConfig:
    target: str = "Label"

    # Columns that behave like identifiers or show near-perfect target purity.
    leakage_candidates: List[str] = field(
        default_factory=lambda: [
            "Variant_ID",
            "AL_1",
            "AL_2",
            "AL_3",
            "AL_4",
            "AL_5",
            "AL_6",
            "AL_28",
            "AL_30",
            "AL_31",
            "AL_32",
            "AL_33",
            "AL_34",
            "AL_35",
            "AL_36",
            "AL_37",
            "AL_38",
            "AL_301",
        ]
    )

    # Very high missingness columns observed in EDA.
    high_missing_candidates: List[str] = field(
        default_factory=lambda: [
            "CAT_6",
            "AL_1",
            "AL_2",
            "AL_3",
            "AL_4",
            "AL_5",
            "AL_6",
            "AL_27",
            "AL_28",
            "AL_29",
            "AL_30",
            "AL_31",
            "AL_32",
            "AL_33",
            "AL_34",
            "AL_35",
        ]
    )

    # Feature groups by prefix.
    prefix_groups: Dict[str, str] = field(
        default_factory=lambda: {
            "AL": "allele",
            "CAT": "category",
            "EK": "experimental",
            "AA": "amino",
        }
    )


@dataclass(frozen=True)
class TrainingConfig:
    paths: DatasetPaths
    features: FeatureConfig
    categorical_max_cardinality: int = 20
    high_missing_threshold: float = 0.95


def default_config(root: str | Path) -> TrainingConfig:
    root_path = Path(root)
    return TrainingConfig(paths=DatasetPaths(root_path), features=FeatureConfig())
