from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score


def binary_metrics(y_true, probability, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": int(len(y)), "threshold": float(threshold), "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "specificity": specificity,
        "f1": f1_score(y, pred, zero_division=0), "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0), "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "log_loss": log_loss(y, p, labels=[0, 1]), "brier_score": brier_score_loss(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_true, probability, objective: str) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    metric = "f1_macro" if objective == "f1_macro" else "mcc"
    scores = [(binary_metrics(y_true, probability, float(t))[metric], float(t)) for t in candidates]
    return max(scores, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]
