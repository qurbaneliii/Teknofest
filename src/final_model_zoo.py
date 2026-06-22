from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from medical_metrics import compute_medical_metrics
from teknofest.data_prep import PreparedData
from teknofest.features import FeatureEngineer, detect_binary_al_cols
from teknofest.training import align_numeric, make_lgbm, model_columns
from teknofest.validation import contamination_aware_folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAVED_FINAL_OOF = PREDICTIONS_DIR / "final_master_cv_predictions.csv"
SAVED_FINAL_PANEL = PREDICTIONS_DIR / "final_panel_predictions.csv"
SAVED_FINAL_CONFIG = PROJECT_ROOT / "artifacts" / "metrics" / "final_config.json"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    params: dict[str, Any]
    is_saved_reference: bool = False


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("lightgbm_conservative_regularized", "lightgbm_reference", {}, True),
        ModelSpec(
            "lightgbm_high_capacity_controlled",
            "lightgbm",
            {
                "n_estimators": 350,
                "learning_rate": 0.03,
                "num_leaves": 127,
                "max_depth": 8,
                "min_child_samples": 25,
                "min_split_gain": 0.0,
                "subsample": 0.85,
                "subsample_freq": 1,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.1,
                "reg_lambda": 2.0,
                "max_bin": 255,
                "n_jobs": 4,
            },
        ),
        ModelSpec("catboost", "catboost", {"iterations": 200, "depth": 6, "learning_rate": 0.05}),
        ModelSpec("xgboost", "xgboost", {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.05}),
        ModelSpec("extra_trees", "extra_trees", {"n_estimators": 250, "min_samples_leaf": 2}),
        ModelSpec("elasticnet_logistic_regression", "elasticnet_logistic", {"C": 0.35, "l1_ratio": 0.25}),
    ]


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Required prediction file {source} is missing columns: {sorted(missing)}")


def _saved_reference_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SAVED_FINAL_OOF.exists():
        raise FileNotFoundError(f"Saved final OOF predictions are required for the LightGBM reference: {SAVED_FINAL_OOF}")
    if not SAVED_FINAL_PANEL.exists():
        raise FileNotFoundError(f"Saved final panel predictions are required for the LightGBM reference: {SAVED_FINAL_PANEL}")
    oof = pd.read_csv(SAVED_FINAL_OOF)
    panel = pd.read_csv(SAVED_FINAL_PANEL)
    _require_columns(oof, {"fold", "row_index", "Variant_ID", "Label", "score", "threshold"}, SAVED_FINAL_OOF)
    _require_columns(panel, {"dataset", "Variant_ID", "Label", "score", "threshold"}, SAVED_FINAL_PANEL)
    if oof.empty or panel.empty:
        raise ValueError("Saved final reference prediction files must contain rows.")
    return oof, panel


def _conservative_base_parameters() -> dict[str, Any]:
    if not SAVED_FINAL_CONFIG.exists():
        raise FileNotFoundError(f"Saved final model configuration is missing: {SAVED_FINAL_CONFIG}")
    config = json.loads(SAVED_FINAL_CONFIG.read_text(encoding="utf-8"))
    params = config.get("selected_lightgbm_config")
    if not isinstance(params, dict):
        raise ValueError("Saved final model configuration has no selected_lightgbm_config dictionary.")
    return dict(params)


def _fold_engineered_data(prepared: PreparedData, train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flags)
    train = engineer.fit_transform(prepared.master.iloc[train_idx].copy())
    validation = engineer.transform(prepared.master.iloc[val_idx].copy())
    return align_numeric(train, validation)


def _pipeline(estimator: Any, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _make_model(spec: ModelSpec, lgbm_base_params: dict[str, Any]) -> Any:
    if spec.family == "lightgbm":
        params = {**lgbm_base_params, **spec.params}
        return make_lgbm(params)
    if spec.family == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
            thread_count=4,
            l2_leaf_reg=5.0,
            **spec.params,
        )
    if spec.family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_alpha=0.2,
            reg_lambda=4.0,
            **spec.params,
        )
    if spec.family == "extra_trees":
        return _pipeline(
            ExtraTreesClassifier(
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=4,
                **spec.params,
            )
        )
    if spec.family == "elasticnet_logistic":
        return _pipeline(
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=spec.params["C"],
                l1_ratio=spec.params["l1_ratio"],
                class_weight="balanced",
                max_iter=500,
                n_jobs=4,
                random_state=42,
            ),
            scale=True,
        )
    raise ValueError(f"Unsupported model family: {spec.family}")


def _fit_predict(spec: ModelSpec, x_train: pd.DataFrame, y_train: pd.Series, x_eval: pd.DataFrame, lgbm_base_params: dict[str, Any]) -> tuple[Any, np.ndarray]:
    model = _make_model(spec, lgbm_base_params)
    model.fit(x_train, y_train)
    return model, np.asarray(model.predict_proba(x_eval)[:, 1], dtype=float)


def _panel_raw(prepared: PreparedData) -> pd.DataFrame:
    frames = []
    for dataset, raw in (
        ("KANSER_unique", prepared.kanser_unique),
        ("PAH_unique", prepared.pah_unique),
        ("CFTR_unique", prepared.cftr_unique),
    ):
        copy = raw.copy()
        copy["dataset"] = dataset
        frames.append(copy)
    return pd.concat(frames, ignore_index=True)


def _fit_full_and_predict_panel(spec: ModelSpec, prepared: PreparedData, panel_raw: pd.DataFrame, lgbm_base_params: dict[str, Any]) -> np.ndarray:
    flags = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flags)
    train = engineer.fit_transform(prepared.master.copy())
    panel = engineer.transform(panel_raw.copy())
    columns = model_columns(train)
    model, _ = _fit_predict(spec, train[columns], train["Label"].astype(int), train[columns], lgbm_base_params)
    return np.asarray(model.predict_proba(panel.reindex(columns=columns))[:, 1], dtype=float)


def _metric_row(spec: ModelSpec, y: pd.Series, probability: pd.Series, threshold: float, status: str, failure_reason: str = "") -> dict[str, object]:
    metrics = compute_medical_metrics(y, probability, threshold)
    metrics.update(
        {
            "model_id": spec.model_id,
            "model_family": spec.family,
            "threshold_source": "saved_final_threshold",
            "status": status,
            "failure_reason": failure_reason,
        }
    )
    return metrics


def _failure_row(spec: ModelSpec, failure_reason: str) -> dict[str, object]:
    return {
        "model_id": spec.model_id,
        "model_family": spec.family,
        "threshold_source": "saved_final_threshold",
        "status": "failed",
        "failure_reason": failure_reason,
    }


def _summary(metrics: pd.DataFrame, threshold: float) -> str:
    successful = metrics[metrics["status"].eq("success")].copy()
    lines = [
        "# Model Zoo Summary",
        "",
        "All challenger models use the existing fold-safe FeatureEngineer and contamination-aware folds. The LightGBM conservative reference is copied from the immutable saved Phase 10 OOF artifact, so its result is directly reproducible without altering that model.",
        "",
        f"All decision metrics use the unchanged saved final threshold `{threshold:.6f}`. No model is selected as final in this phase.",
        "",
    ]
    if successful.empty:
        lines.append("No model-zoo run completed successfully.")
    else:
        columns = ["model_id", "roc_auc", "pr_auc", "f1_macro", "mcc", "medical_utility_score", "pathogenic_recall", "specificity"]
        lines.extend(["## OOF Metrics", "", successful[columns].sort_values("medical_utility_score", ascending=False).to_markdown(index=False), ""])
        lines.extend(["## Per-Metric Leaders", ""])
        for metric, label in (
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
            ("f1_macro", "F1-macro"),
            ("mcc", "MCC"),
            ("medical_utility_score", "MedicalUtilityScore"),
        ):
            row = successful.loc[successful[metric].idxmax()]
            lines.append(f"- {label}: `{row['model_id']}` ({float(row[metric]):.6f})")
    failures = metrics[metrics["status"].eq("failed")]
    if not failures.empty:
        lines.extend(["", "## Rejected/Failed Runs", "", failures[["model_id", "failure_reason"]].to_markdown(index=False)])
    return "\n".join(lines) + "\n"


def run_final_model_zoo(prepared: PreparedData, specs: list[ModelSpec] | None = None) -> dict[str, pd.DataFrame]:
    """Evaluate the Phase 2 model zoo without changing final-model artifacts."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fold_output_dir = PREDICTIONS_DIR / "model_zoo_folds"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or default_model_specs()
    reference_oof, reference_panel = _saved_reference_predictions()
    threshold_values = reference_oof["threshold"].drop_duplicates()
    if len(threshold_values) != 1:
        raise ValueError("Saved final OOF predictions must contain one immutable threshold.")
    threshold = float(threshold_values.iloc[0])
    lgbm_base_params = _conservative_base_parameters()

    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    valid_indices = np.concatenate([fold.val_idx for fold in folds])
    oof = reference_oof[["fold", "row_index", "Variant_ID", "Label"]].copy().sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(np.sort(oof["row_index"].to_numpy()), np.sort(valid_indices)):
        raise ValueError("Saved final OOF rows do not match the current contamination-aware validation folds.")
    oof["proba__lightgbm_conservative_regularized"] = reference_oof.set_index("row_index").loc[oof["row_index"], "score"].to_numpy()
    panel = reference_panel[["dataset", "Variant_ID", "Label"]].copy()
    panel["proba__lightgbm_conservative_regularized"] = reference_panel["score"].to_numpy()

    metric_rows = [
        _metric_row(
            specs[0],
            oof["Label"],
            oof["proba__lightgbm_conservative_regularized"],
            threshold,
            "success",
        )
    ]
    fold_rows: list[dict[str, object]] = []
    for fold, frame in oof.groupby("fold"):
        row = _metric_row(specs[0], frame["Label"], frame["proba__lightgbm_conservative_regularized"], threshold, "success")
        row["fold"] = int(fold)
        fold_rows.append(row)

    panel_raw = _panel_raw(prepared)
    for spec in specs[1:]:
        print(f"[model-zoo] training {spec.model_id}", flush=True)
        column = f"proba__{spec.model_id}"
        oof[column] = np.nan
        fold_failure: str | None = None
        for fold in folds:
            try:
                x_train, x_val = _fold_engineered_data(prepared, fold.train_idx, fold.val_idx)
                y_train = prepared.master.iloc[fold.train_idx]["Label"].astype(int)
                _, probability = _fit_predict(spec, x_train, y_train, x_val, lgbm_base_params)
                oof.loc[oof["row_index"].isin(fold.val_idx), column] = probability
            except Exception as exc:  # Keep the model-zoo run alive when an optional dependency fails.
                fold_failure = f"fold {fold.fold}: {type(exc).__name__}: {exc}"
                break
        if fold_failure or oof[column].isna().any():
            oof.drop(columns=column, inplace=True)
            metric_rows.append(_failure_row(spec, fold_failure or "OOF prediction matrix was incomplete."))
            print(f"[model-zoo] {spec.model_id} failed: {metric_rows[-1]['failure_reason']}", flush=True)
            continue

        metric_rows.append(_metric_row(spec, oof["Label"], oof[column], threshold, "success"))
        for fold, frame in oof.groupby("fold"):
            row = _metric_row(spec, frame["Label"], frame[column], threshold, "success")
            row["fold"] = int(fold)
            fold_rows.append(row)
        try:
            panel[column] = _fit_full_and_predict_panel(spec, prepared, panel_raw, lgbm_base_params)
            print(f"[model-zoo] {spec.model_id} complete", flush=True)
        except Exception as exc:
            panel[column] = np.nan
            metric_rows[-1]["panel_prediction_status"] = "failed"
            metric_rows[-1]["panel_prediction_failure"] = f"{type(exc).__name__}: {exc}"

    for row in metric_rows:
        if row["status"] != "success":
            continue
        column = f"proba__{row['model_id']}"
        if column not in panel.columns or panel[column].isna().any():
            row["panel_prediction_status"] = row.get("panel_prediction_status", "unavailable")
            continue
        panel_metrics = compute_medical_metrics(panel["Label"], panel[column], threshold)
        row.update({f"panel_{name}": value for name, value in panel_metrics.items()})
        row["panel_prediction_status"] = "success"

    for fold, frame in oof.groupby("fold"):
        fold_frame = frame.copy()
        for column in fold_frame.filter(like="proba__"):
            fold_frame[f"prediction__{column.removeprefix('proba__')}"] = (fold_frame[column] >= threshold).astype(int)
        fold_frame.to_csv(fold_output_dir / f"fold_{int(fold)}_predictions.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    oof.to_csv(PREDICTIONS_DIR / "model_zoo_oof_predictions.csv", index=False)
    panel.to_csv(PREDICTIONS_DIR / "model_zoo_panel_predictions.csv", index=False)
    metrics.to_csv(TABLES_DIR / "model_zoo_metrics.csv", index=False)
    fold_metrics.to_csv(TABLES_DIR / "model_zoo_fold_metrics.csv", index=False)
    (REPORTS_DIR / "model_zoo_summary.md").write_text(_summary(metrics, threshold), encoding="utf-8")
    return {"oof_predictions": oof, "panel_predictions": panel, "metrics": metrics, "fold_metrics": fold_metrics}
