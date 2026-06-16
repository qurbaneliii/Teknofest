from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DataConfig:
    root: Path
    train_files: List[str] = field(
        default_factory=lambda: [
            "YARISMA_TRAIN_CFTR.csv",
            "YARISMA_TRAIN_KANSER.csv",
            "YARISMA_TRAIN_MASTER.csv",
            "YARISMA_TRAIN_PAH.csv",
        ]
    )
    target: str = "Label"
    dataset_id_col: str = "dataset_id"


@dataclass(frozen=True)
class LeakageConfig:
    high_cardinality_threshold: float = 0.95
    purity_threshold: float = 0.98
    permutation_importance_threshold: float = 0.001
    batch_effect_mi_threshold: float = 0.01
    max_low_cardinality: int = 50


@dataclass(frozen=True)
class FeatureConfig:
    categorical_max_cardinality: int = 20
    hashing_bins: int = 128
    target_encoding_smoothing: float = 10.0
    interaction_top_k: int = 20
    missing_block_prefixes: List[str] = field(
        default_factory=lambda: ["AL_", "EK_", "AA_", "CAT_"]
    )


@dataclass(frozen=True)
class SelectionConfig:
    mi_top_k: int = 200
    stability_threshold: float = 0.7
    permutation_top_k: int = 200


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    random_state: int = 42
    use_group_kfold: bool = True
    use_dataset_holdout: bool = True


@dataclass(frozen=True)
class ModelConfig:
    logistic_params: Dict[str, object] = field(
        default_factory=lambda: {"C": 1.0, "max_iter": 500, "class_weight": "balanced"}
    )
    catboost_params: Dict[str, object] = field(
        default_factory=lambda: {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "random_seed": 42,
            "verbose": False,
        }
    )
    lightgbm_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
        }
    )


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig
    leakage: LeakageConfig = LeakageConfig()
    features: FeatureConfig = FeatureConfig()
    selection: SelectionConfig = SelectionConfig()
    cv: CVConfig = CVConfig()
    models: ModelConfig = ModelConfig()
    output_dir: Path = Path("artifacts")


def default_config(root: str | Path) -> PipelineConfig:
    return PipelineConfig(data=DataConfig(root=Path(root)))
