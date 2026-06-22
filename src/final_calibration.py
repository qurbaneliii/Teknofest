from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from medical_metrics import compute_medical_metrics


EPSILON = 1e-6


def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0 - EPSILON)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    prob = _clip(probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (prob >= left) & ((prob < right) if right < 1 else (prob <= right))
        if not mask.any():
            continue
        gap = abs(float(y[mask].mean()) - float(prob[mask].mean()))
        ece += mask.mean() * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def calibration_slope_intercept(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    if np.unique(y).size < 2:
        return np.nan, np.nan
    logits = np.log(_clip(probabilities) / (1.0 - _clip(probabilities))).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=2000, random_state=42).fit(logits, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


@dataclass
class ProbabilityCalibrator:
    method: str
    model: object | None = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        prob = _clip(probabilities)
        y = np.asarray(y_true, dtype=int)
        if self.method == "none":
            self.model = None
        elif self.method == "sigmoid":
            self.model = LogisticRegression(C=1e6, max_iter=3000, random_state=42).fit(prob.reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(prob, y)
        elif self.method == "beta":
            features = np.c_[np.log(prob), np.log(1.0 - prob)]
            self.model = LogisticRegression(C=1e6, max_iter=3000, random_state=42).fit(features, y)
        elif self.method == "temperature":
            logits = np.log(prob / (1.0 - prob))
            objective = lambda log_temperature: -np.mean(
                y * np.log(_clip(expit(logits / np.exp(log_temperature))))
                + (1 - y) * np.log(_clip(1.0 - expit(logits / np.exp(log_temperature))))
            )
            result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
            self.model = float(np.exp(result.x))
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        prob = _clip(probabilities)
        if self.method == "none":
            return prob
        if self.method == "isotonic":
            return _clip(np.asarray(self.model.predict(prob), dtype=float))
        if self.method == "beta":
            features = np.c_[np.log(prob), np.log(1.0 - prob)]
        elif self.method == "temperature":
            return _clip(expit(np.log(prob / (1.0 - prob)) / float(self.model)))
        else:
            features = prob.reshape(-1, 1)
        return _clip(np.asarray(self.model.predict_proba(features)[:, 1], dtype=float))


def cross_fitted_calibration(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    method: str,
    folds: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    prob = _clip(probabilities)
    if method == "none":
        return prob
    if folds is None:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_list = list(splitter.split(prob, y))
    else:
        fold_values = np.asarray(folds)
        split_list = [(np.flatnonzero(fold_values != fold), np.flatnonzero(fold_values == fold)) for fold in np.unique(fold_values)]
    calibrated = np.empty(len(y), dtype=float)
    for train_idx, val_idx in split_list:
        calibrator = ProbabilityCalibrator(method).fit(prob[train_idx], y[train_idx])
        calibrated[val_idx] = calibrator.predict(prob[val_idx])
    return calibrated


def calibration_comparison(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    threshold: float,
    folds: np.ndarray | pd.Series | None = None,
    methods: tuple[str, ...] = ("none", "sigmoid", "isotonic", "beta", "temperature"),
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y = np.asarray(y_true, dtype=int)
    comparison: list[dict[str, object]] = []
    calibrated_by_method: dict[str, np.ndarray] = {}
    for method in methods:
        calibrated = cross_fitted_calibration(y, probabilities, method, folds)
        metric = compute_medical_metrics(y, calibrated, threshold)
        ece, mce = expected_calibration_error(y, calibrated)
        slope, intercept = calibration_slope_intercept(y, calibrated)
        metric.update(
            {
                "calibration_method": method,
                "ece": ece,
                "mce": mce,
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        )
        comparison.append(metric)
        calibrated_by_method[method] = calibrated
    return pd.DataFrame(comparison), calibrated_by_method


def choose_calibration(comparison: pd.DataFrame) -> pd.Series:
    """Accept a calibrator only if probability loss improves without decision harm."""
    baseline = comparison[comparison["calibration_method"].eq("none")].iloc[0]
    candidates = comparison.copy()
    candidates["decision_preserved"] = (
        candidates["mcc"].ge(float(baseline["mcc"]) - 0.015)
        & candidates["f1_macro"].ge(float(baseline["f1_macro"]) - 0.015)
        & candidates["pathogenic_recall"].ge(float(baseline["pathogenic_recall"]) - 0.02)
    )
    acceptable = candidates[candidates["decision_preserved"] & candidates["brier_score"].lt(float(baseline["brier_score"]))]
    if acceptable.empty:
        selected = baseline.copy()
        selected["selection_reason"] = "No calibration method improved Brier score while preserving decision metrics."
        return selected
    selected = acceptable.sort_values(["brier_score", "ece", "mcc"], ascending=[True, True, False]).iloc[0].copy()
    selected["selection_reason"] = "Improved probability calibration while preserving OOF decision metrics."
    return selected


def save_calibration_outputs(
    comparison: pd.DataFrame,
    calibrated_by_method: dict[str, np.ndarray],
    y_true: np.ndarray | pd.Series,
    reports_dir: str | Path = "reports",
) -> pd.Series:
    reports = Path(reports_dir)
    tables = reports / "tables"
    figures = reports / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(tables / "final_calibration_comparison.csv", index=False)
    selected = choose_calibration(comparison)
    y = np.asarray(y_true, dtype=int)

    plt.figure(figsize=(7, 6))
    for method, probabilities in calibrated_by_method.items():
        bins = pd.DataFrame({"y": y, "prob": probabilities}).assign(bin=lambda frame: pd.cut(frame["prob"], bins=np.linspace(0, 1, 11), include_lowest=True))
        reliability = bins.groupby("bin", observed=False).agg(observed=("y", "mean"), predicted=("prob", "mean")).dropna()
        plt.plot(reliability["predicted"], reliability["observed"], marker="o", label=method)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed pathogenic rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "final_reliability_diagram.png", dpi=180)
    plt.savefig(figures / "calibration_before_after.png", dpi=180)
    plt.close()

    decision = [
        "# Final Calibration Decision",
        "",
        f"Selected method: `{selected['calibration_method']}`.",
        "",
        str(selected["selection_reason"]),
        "",
        "Calibration is evaluated with cross-fitted calibrators over OOF predictions. It is not selected solely because Brier loss improves.",
    ]
    (reports / "final_calibration_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    return selected
