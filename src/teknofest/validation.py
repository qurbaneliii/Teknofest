from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class FoldIndices:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def contamination_aware_folds(
    labels: pd.Series,
    shared_mask: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> list[FoldIndices]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldIndices] = []
    row_idx = np.arange(len(labels))

    for fold, (train_idx, val_idx) in enumerate(skf.split(row_idx, labels)):
        val_idx = np.asarray(
            [idx for idx in val_idx if not bool(shared_mask.iloc[idx])],
            dtype=int,
        )
        folds.append(FoldIndices(fold=fold, train_idx=train_idx, val_idx=val_idx))

    return folds


def fold_assignment_frame(
    master: pd.DataFrame,
    shared_mask: pd.Series,
    folds: list[FoldIndices],
) -> pd.DataFrame:
    assignment = pd.DataFrame(
        {
            "row_index": np.arange(len(master)),
            "Variant_ID": master["Variant_ID"].to_numpy(),
            "Label": master["Label"].to_numpy(),
            "is_master_shared": shared_mask.to_numpy(dtype=bool),
            "validation_fold": -1,
        }
    )
    for fold in folds:
        assignment.loc[fold.val_idx, "validation_fold"] = fold.fold
    return assignment


def fold_summary(master: pd.DataFrame, folds: list[FoldIndices]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        y_train = master.iloc[fold.train_idx]["Label"]
        y_val = master.iloc[fold.val_idx]["Label"]
        rows.append(
            {
                "fold": fold.fold,
                "train_n": len(fold.train_idx),
                "train_pathogenic_rate": float(y_train.mean()),
                "val_n": len(fold.val_idx),
                "val_pathogenic_rate": float(y_val.mean()),
                "val_pathogenic": int(y_val.sum()),
                "val_benign": int((y_val == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def best_f1_macro_threshold(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    candidates = np.r_[thresholds, 1.0]
    scores = [
        f1_score(y_true, np.asarray(y_score) >= threshold, average="macro")
        for threshold in candidates
    ]
    best_idx = int(np.argmax(scores))
    return float(candidates[best_idx]), float(scores[best_idx])


def youden_j_threshold(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])
