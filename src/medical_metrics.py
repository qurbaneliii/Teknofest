from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


EPSILON = 1e-7


def _safe_metric(metric, y_true: np.ndarray, values: np.ndarray, default: float = np.nan) -> float:
    try:
        return float(metric(y_true, values))
    except ValueError:
        return float(default)


def calibration_quality(brier_score: float | None) -> float:
    """Map Brier loss to a bounded higher-is-better clinical calibration score."""
    if brier_score is None or not np.isfinite(brier_score):
        return 0.0
    # 0.25 is the Brier loss of an uninformative balanced binary predictor.
    return float(np.clip(1.0 - float(brier_score) / 0.25, 0.0, 1.0))


def medical_utility_score(metrics: dict[str, float] | pd.Series) -> float:
    """Primary competition selection score specified in the final improvement brief."""
    get = metrics.get
    return float(
        0.18 * float(get("roc_auc", 0.0) or 0.0)
        + 0.18 * float(get("pr_auc", 0.0) or 0.0)
        + 0.18 * float(get("f1_macro", 0.0) or 0.0)
        + 0.18 * float(get("mcc", 0.0) or 0.0)
        + 0.12 * float(get("balanced_accuracy", 0.0) or 0.0)
        + 0.10 * float(get("pathogenic_recall", get("recall", 0.0)) or 0.0)
        + 0.06 * float(get("specificity", 0.0) or 0.0)
    )


def clinical_safety_score(metrics: dict[str, float] | pd.Series) -> float:
    """Clinical-priority score that gives more weight to pathogenic sensitivity."""
    get = metrics.get
    return float(
        0.25 * float(get("pathogenic_recall", get("recall", 0.0)) or 0.0)
        + 0.20 * float(get("pr_auc", 0.0) or 0.0)
        + 0.20 * float(get("mcc", 0.0) or 0.0)
        + 0.15 * float(get("f1_macro", 0.0) or 0.0)
        + 0.10 * float(get("specificity", 0.0) or 0.0)
        + 0.10 * float(get("calibration_quality", 0.0) or 0.0)
    )


@dataclass(frozen=True)
class MedicalMetrics:
    threshold: float
    roc_auc: float
    pr_auc: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    pathogenic_recall: float
    specificity: float
    f1_binary: float
    f1_macro: float
    f1_weighted: float
    mcc: float
    cohen_kappa: float
    brier_score: float
    log_loss: float
    ppv: float
    npv: float
    tn: int
    fp: int
    fn: int
    tp: int
    false_negative_rate: float
    false_positive_rate: float
    calibration_quality: float
    medical_utility_score: float
    clinical_safety_score: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_medical_metrics(
    y_true: Iterable[int] | np.ndarray | pd.Series,
    probabilities: Iterable[float] | np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute threshold-free and threshold-dependent clinical binary metrics.

    Class ``1`` is pathogenic and class ``0`` is benign.  The function accepts
    a single-class slice as well; ranking metrics are then reported as NaN
    rather than silently fabricated.
    """
    y = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0 - EPSILON)
    if y.ndim != 1 or prob.ndim != 1 or len(y) != len(prob):
        raise ValueError("y_true and probabilities must be one-dimensional arrays of equal length.")
    if len(y) == 0:
        raise ValueError("Medical metrics require at least one observation.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true must contain binary benign (0) / pathogenic (1) labels.")

    pred = (prob >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    pathogenic_recall = float(tp / (tp + fn)) if tp + fn else 0.0
    ppv = float(tp / (tp + fp)) if tp + fp else 0.0
    npv = float(tn / (tn + fn)) if tn + fn else 0.0
    brier = float(brier_score_loss(y, prob))

    values: dict[str, float | int] = {
        "threshold": float(threshold),
        "roc_auc": _safe_metric(roc_auc_score, y, prob),
        "pr_auc": _safe_metric(average_precision_score, y, prob),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "pathogenic_recall": pathogenic_recall,
        "specificity": specificity,
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_binary": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "brier_score": brier,
        "log_loss": _safe_metric(log_loss, y, prob),
        "ppv": ppv,
        "npv": npv,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "calibration_quality": calibration_quality(brier),
    }
    values["medical_utility_score"] = medical_utility_score(values)
    values["clinical_safety_score"] = clinical_safety_score(values)
    return values


def aggregate_fold_metrics(
    fold_metrics: pd.DataFrame,
    metric_columns: list[str] | None = None,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize per-fold metrics with reproducible bootstrap CIs for the mean."""
    if fold_metrics.empty:
        return pd.DataFrame(columns=["metric", "mean", "std", "min", "max", "ci95_low", "ci95_high", "n_folds"])
    excluded = {"fold", "threshold", "tn", "fp", "fn", "tp"}
    columns = metric_columns or [
        col for col in fold_metrics.columns if col not in excluded and pd.api.types.is_numeric_dtype(fold_metrics[col])
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for column in columns:
        values = pd.to_numeric(fold_metrics[column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        samples = rng.choice(values, size=(bootstrap_iterations, len(values)), replace=True).mean(axis=1)
        rows.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
                "n_folds": int(len(values)),
            }
        )
    return pd.DataFrame(rows)
