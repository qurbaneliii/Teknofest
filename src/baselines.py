from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

from teknofest.data_prep import PreparedData
from teknofest.training import acmg_rule_probability, fit_lr_ek, fold_engineered_data
from teknofest.validation import contamination_aware_folds


def _metric_record(model: str, fold: int, y_true, y_score, threshold: float = 0.5) -> dict[str, object]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model,
        "fold": fold,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_baselines(prepared: PreparedData) -> pd.DataFrame:
    rows = []
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    ek_cols = [f"EK_{i}" for i in range(1, 10)]
    for fold in folds:
        train_df, val_df = fold_engineered_data(prepared, fold.train_idx, fold.val_idx)
        y_train = train_df["Label"]
        y_val = val_df["Label"]

        majority_score = pd.Series(float(y_train.mean() >= 0.5), index=val_df.index)
        rows.append(_metric_record("majority_class", fold.fold, y_val, majority_score))

        acmg_score = acmg_rule_probability(val_df)
        rows.append(_metric_record("acmg_rule", fold.fold, y_val, acmg_score))

        lr = fit_lr_ek()
        lr.fit(train_df[ek_cols], y_train)
        lr_score = lr.predict_proba(val_df[ek_cols])[:, 1]
        rows.append(_metric_record("lr_ek_only", fold.fold, y_val, lr_score))
    return pd.DataFrame(rows)

