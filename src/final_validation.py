from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from advanced_bio_features import AdvancedBioFeatureEngineer
from final_model_zoo import (
    _fit_predict,
    _fold_frames,
    _model_matrix,
    _panel_frame,
    _select_columns,
    default_model_specs,
)
from final_thresholding import select_threshold_candidates, threshold_grid
from medical_metrics import compute_medical_metrics
from teknofest.data_prep import PreparedData
from teknofest.features import FeatureEngineer, detect_binary_al_cols
from teknofest.validation import contamination_aware_folds


DEFAULT_SEEDS = (13, 21, 42, 77, 101)


def _medical_threshold(y: pd.Series, probabilities: np.ndarray) -> float:
    candidates = select_threshold_candidates(threshold_grid(y, probabilities))
    row = candidates[candidates["threshold_strategy"].eq("max_medical_utility")].iloc[0]
    return float(row["threshold"])


def _contamination_oof_threshold(prepared: PreparedData, spec) -> float:
    predictions: list[pd.DataFrame] = []
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask, n_splits=5, seed=42)
    for fold in folds:
        train, validation, _, _ = _fold_frames(prepared, fold.train_idx, fold.val_idx)
        columns, _ = _select_columns(train, spec)
        _, probability = _fit_predict(
            spec,
            _model_matrix(train, columns),
            train["Label"].astype(int),
            _model_matrix(validation, columns),
        )
        predictions.append(pd.DataFrame({"Label": validation["Label"].astype(int), "probability": probability}))
    oof = pd.concat(predictions, ignore_index=True)
    return _medical_threshold(oof["Label"], oof["probability"].to_numpy())


def repeated_contamination_aware_validation(
    prepared: PreparedData,
    model_id: str = "lightgbm_conservative_regularized",
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {spec.model_id: spec for spec in default_model_specs()}
    if model_id not in specs:
        raise ValueError(f"Unknown model id: {model_id}")
    spec = specs[model_id]
    seed_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for seed in seeds:
        folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask, n_splits=5, seed=seed)
        predictions: list[pd.DataFrame] = []
        for fold in folds:
            train, validation, _, _ = _fold_frames(prepared, fold.train_idx, fold.val_idx)
            columns, _ = _select_columns(train, spec)
            _, probability = _fit_predict(
                spec,
                _model_matrix(train, columns),
                train["Label"].astype(int),
                _model_matrix(validation, columns),
            )
            predictions.append(
                pd.DataFrame({"fold": fold.fold, "Label": validation["Label"].astype(int), "probability": probability})
            )
        seed_oof = pd.concat(predictions, ignore_index=True)
        threshold = _medical_threshold(seed_oof["Label"], seed_oof["probability"].to_numpy())
        seed_metric = compute_medical_metrics(seed_oof["Label"], seed_oof["probability"], threshold)
        seed_metric.update({"model_id": model_id, "seed": seed, "threshold": threshold})
        seed_rows.append(seed_metric)
        for fold, frame in seed_oof.groupby("fold"):
            metric = compute_medical_metrics(frame["Label"], frame["probability"], threshold)
            metric.update({"model_id": model_id, "seed": seed, "fold": int(fold), "threshold": threshold})
            fold_rows.append(metric)
    return pd.DataFrame(seed_rows), pd.DataFrame(fold_rows)


def panel_and_stress_validation(prepared: PreparedData, model_id: str = "lightgbm_conservative_regularized") -> pd.DataFrame:
    specs = {spec.model_id: spec for spec in default_model_specs()}
    spec = specs[model_id]
    raw_panel = _panel_frame(prepared)
    flags = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flags)
    advanced = AdvancedBioFeatureEngineer()
    train = advanced.transform(engineer.fit_transform(prepared.master.copy()))
    panel = advanced.transform(engineer.transform(raw_panel))
    policies = {
        "MASTER_only_full_safe": spec,
        "MASTER_only_compact_stable": replace(spec, feature_set="compact_stable_features"),
        "MASTER_only_no_target_encoded": replace(spec, feature_set="no_target_encoded_features"),
    }
    rows: list[dict[str, object]] = []
    for policy, candidate in policies.items():
        columns, _ = _select_columns(train, candidate)
        model, probabilities = _fit_predict(
            candidate,
            _model_matrix(train, columns),
            train["Label"].astype(int),
            _model_matrix(panel, columns),
        )
        del model
        threshold = _contamination_oof_threshold(prepared, candidate)
        for panel_name, frame in raw_panel.assign(probability=probabilities).groupby("panel"):
            metric = compute_medical_metrics(frame["Label"], frame["probability"], threshold)
            metric.update({"model_id": model_id, "stress_policy": policy, "evaluation_split": panel_name, "threshold": threshold})
            rows.append(metric)
        combined = compute_medical_metrics(raw_panel["Label"], probabilities, threshold)
        combined.update({"model_id": model_id, "stress_policy": policy, "evaluation_split": "panel_unique_combined", "threshold": threshold})
        rows.append(combined)
    return pd.DataFrame(rows)


def save_final_validation_outputs(
    seed_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    stress_metrics: pd.DataFrame,
    reports_dir: str | Path = "reports",
) -> None:
    reports = Path(reports_dir)
    tables = reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(tables / "repeated_cv_metrics.csv", index=False)
    stability = seed_metrics.groupby("model_id", as_index=False).agg(
        medical_utility_mean=("medical_utility_score", "mean"),
        medical_utility_std=("medical_utility_score", "std"),
        mcc_mean=("mcc", "mean"),
        mcc_std=("mcc", "std"),
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
        threshold_std=("threshold", "std"),
    )
    stability.to_csv(tables / "seed_stability_metrics.csv", index=False)
    stress_metrics.to_csv(tables / "validation_stress_test_results.csv", index=False)
    stress_metrics[stress_metrics["stress_policy"].eq("MASTER_only_full_safe")].to_csv(
        tables / "panel_unique_final_metrics.csv", index=False
    )
    text = """# Final Validation Strategy

The final validation protocol uses contamination-aware 5-fold stratification, repeated with seeds 13, 21, 42, 77, and 101. MASTER variants shared with any panel are never validation rows. Panel-unique KANSER, PAH, and CFTR rows are evaluated separately and in combination. Stress checks compare the full safe feature set with compact and no-target-encoding variants.

Leave-one-panel-out model fitting is intentionally not used as a primary score: panels have different disease distributions and their labels are retained as external generalization evidence rather than being mixed into MASTER training.
"""
    (reports / "final_validation_strategy.md").write_text(text, encoding="utf-8")
