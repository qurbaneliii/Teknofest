from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import stats
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from teknofest.data_prep import PreparedData
from teknofest.features import FeatureEngineer, detect_binary_al_cols
from teknofest.training import (
    acmg_rule_probability,
    align_numeric,
    fit_final_lgbm,
    fit_lr_ek,
    make_extra_trees,
    make_lgbm,
    model_columns,
    threshold_metric_rows,
)
from teknofest.validation import (
    best_f1_macro_threshold,
    contamination_aware_folds,
    youden_j_threshold,
)


ABLATION_DEFINITIONS = {
    "ABL-01_EK_cols_only": "EK raw columns only",
    "ABL-02_AL_cols_only": "AL raw columns only",
    "ABL-03_engineered_only_no_raw_AL_EK": "Engineered features only, excluding raw AL/EK",
    "ABL-04_all_zero_impute_no_miss_flags": "All numeric features, zero imputation, missingness flags removed",
    "ABL-05_all_with_miss_flags": "Full proposed feature set",
    "ABL-06_no_EK_interactions": "Full feature set without EK interaction terms",
    "ABL-07_no_AA_chemistry": "Full feature set without amino-acid chemistry features",
    "ABL-08_no_CAT1_decomposition": "Full feature set without CAT_1 decomposition features",
    "ABL-09_single_LGBM_vs_extra_trees": "Single LightGBM compared with ExtraTrees baseline",
    "ABL-10_default_vs_optimized_threshold": "Default 0.5 threshold vs optimized F1 threshold",
}


def summarize_cv(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc_roc", "auc_pr", "f1_macro", "f1_weighted", "mcc", "cohen_kappa"]
    return (
        results.groupby(["model", "threshold_name"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rows = []
    metrics = {
        "auc_roc": lambda yt, ys, yp: roc_auc_score(yt, ys),
        "auc_pr": lambda yt, ys, yp: average_precision_score(yt, ys),
        "f1_macro": lambda yt, ys, yp: f1_score(yt, yp, average="macro"),
        "f1_weighted": lambda yt, ys, yp: f1_score(yt, yp, average="weighted"),
        "mcc": lambda yt, ys, yp: matthews_corrcoef(yt, yp),
        "cohen_kappa": lambda yt, ys, yp: cohen_kappa_score(yt, yp),
    }
    values = {name: [] for name in metrics}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        ys = y_score[idx]
        yp = (ys >= threshold).astype(int)
        for name, fn in metrics.items():
            values[name].append(float(fn(yt, ys, yp)))
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "metric": name,
                "mean": float(np.mean(arr)),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "n_bootstrap_used": int(len(arr)),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
    name: str,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ece = expected_calibration_error(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))
    table = pd.DataFrame(
        {
            "mean_predicted_probability": mean_pred,
            "fraction_positive": frac_pos,
        }
    )
    table["ece"] = ece
    table["brier_score"] = brier
    table.to_csv(out_dir / f"{name}_calibration.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction positive")
    plt.title(f"Calibration curve ({name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_calibration_curve.png", dpi=170)
    plt.close()
    return table


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10) if (b01 + b10) else 0.0
    p_value = float(stats.chi2.sf(stat, 1))
    return {"b01_a_correct_b_wrong": b01, "b10_a_wrong_b_correct": b10, "statistic": stat, "p_value": p_value}


def delong_placeholder(full_auc: float, baseline_auc: float) -> dict[str, object]:
    return {
        "test": "DeLong",
        "status": "not_computed",
        "reason": "A full DeLong implementation is not available in the current dependency set.",
        "full_model_auc": full_auc,
        "baseline_auc": baseline_auc,
        "auc_delta": full_auc - baseline_auc,
    }


def compute_midrank(x: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(x)
    z = x[sorted_idx]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=float)
    result[sorted_idx] = midranks
    return result


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive[r, :])
        ty[r, :] = compute_midrank(negative[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, np.atleast_2d(delong_cov)


def delong_roc_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = np.vstack([pred_a, pred_b])[:, order]
    aucs, cov = fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = abs(diff) / np.sqrt(var) if var > 0 else np.inf
    p_value = float(2 * stats.norm.sf(z))
    return {
        "test": "DeLong full LightGBM vs EK-only LR",
        "status": "computed",
        "full_model_auc": float(aucs[0]),
        "baseline_auc": float(aucs[1]),
        "auc_delta": float(diff),
        "z_statistic": float(z),
        "p_value": p_value,
    }


def run_l0_stack_oof(
    prepared: PreparedData,
    lgbm_params: dict[str, object] | None,
    out_dir: Path,
    n_estimators: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    params = dict(lgbm_params or {})
    params["n_estimators"] = min(int(params.get("n_estimators", n_estimators)), n_estimators)
    oof_rows = []
    fold_rows = []

    for fold in folds:
        engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
        train = engineer.fit_transform(prepared.master.iloc[fold.train_idx].copy())
        val = engineer.transform(prepared.master.iloc[fold.val_idx].copy())
        y_train = train["Label"]
        y_val = val["Label"]
        x_train, x_val = align_numeric(train, val)
        x_train_filled = x_train.fillna(-999)
        x_val_filled = x_val.fillna(-999)

        lgbm = make_lgbm(params)
        lgbm.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="auc")
        lgbm_prob = lgbm.predict_proba(x_val)[:, 1]

        cat = CatBoostClassifier(
            iterations=n_estimators,
            learning_rate=0.04,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        cat.fit(x_train_filled, y_train)
        cat_prob = cat.predict_proba(x_val_filled)[:, 1]

        xgb = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        xgb.fit(x_train_filled, y_train)
        xgb_prob = xgb.predict_proba(x_val_filled)[:, 1]

        et = make_extra_trees()
        et.set_params(n_estimators=max(100, n_estimators))
        et.fit(x_train_filled, y_train)
        et_prob = et.predict_proba(x_val_filled)[:, 1]

        ek_cols = [f"EK_{i}" for i in range(1, 10)]
        lr = fit_lr_ek()
        lr.fit(train[ek_cols], y_train)
        lr_prob = lr.predict_proba(val[ek_cols])[:, 1]

        probs = {
            "lightgbm": lgbm_prob,
            "catboost": cat_prob,
            "xgboost": xgb_prob,
            "extra_trees": et_prob,
            "lr_ek_only": lr_prob,
        }
        for model_name, prob in probs.items():
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold.fold,
                    "auc_roc": roc_auc_score(y_val, prob),
                    "f1_macro_default_0.5": f1_score(y_val, prob >= 0.5, average="macro"),
                }
            )
        for i, idx in enumerate(fold.val_idx):
            oof_rows.append(
                {
                    "fold": fold.fold,
                    "row_index": idx,
                    "Variant_ID": val["Variant_ID"].iloc[i],
                    "Label": int(y_val.iloc[i]),
                    "EK_7": val["EK_7"].iloc[i],
                    "EK_9": val["EK_9"].iloc[i],
                    "n_pops": val["n_pops"].iloc[i],
                    "max_AF": val["max_AF"].iloc[i],
                    **{f"{name}_probability": float(prob[i]) for name, prob in probs.items()},
                }
            )

    oof = pd.DataFrame(oof_rows)
    meta_cols = [
        "lightgbm_probability",
        "catboost_probability",
        "xgboost_probability",
        "extra_trees_probability",
        "lr_ek_only_probability",
        "EK_7",
        "EK_9",
        "n_pops",
        "max_AF",
    ]
    meta_x = oof[meta_cols].fillna(oof[meta_cols].median(numeric_only=True))
    meta_y = oof["Label"]
    meta = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
    stack_prob = np.zeros(len(oof), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(meta_x, meta_y):
        meta.fit(meta_x.iloc[train_idx], meta_y.iloc[train_idx])
        stack_prob[val_idx] = meta.predict_proba(meta_x.iloc[val_idx])[:, 1]
    oof["stack_probability"] = stack_prob
    fold_rows.append(
        {
            "model": "stack_l1_logreg",
            "fold": -1,
            "auc_roc": roc_auc_score(meta_y, stack_prob),
            "f1_macro_default_0.5": f1_score(meta_y, stack_prob >= 0.5, average="macro"),
        }
    )

    fold_df = pd.DataFrame(fold_rows)
    summary = fold_df.groupby("model")[["auc_roc", "f1_macro_default_0.5"]].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / "l0_stack_oof_predictions.csv", index=False)
    fold_df.to_csv(out_dir / "l0_model_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "l0_model_summary.csv", index=False)
    return oof, fold_df, summary


def feature_subset_columns(df: pd.DataFrame, ablation: str) -> list[str]:
    cols = model_columns(df)
    raw_al = [c for c in cols if c.startswith("AL_")]
    raw_ek = [c for c in cols if c.startswith("EK_") and "x" not in c]
    engineered = [c for c in cols if c not in raw_al and c not in raw_ek]
    if ablation.startswith("ABL-01"):
        return raw_ek
    if ablation.startswith("ABL-02"):
        return raw_al
    if ablation.startswith("ABL-03"):
        return engineered
    if ablation.startswith("ABL-04"):
        return [c for c in cols if not c.startswith("miss_")]
    if ablation.startswith("ABL-06"):
        return [c for c in cols if c not in {"EK7xEK9", "EK2xEK4", "EK4xEK6", "EK7_minus_EK9"}]
    if ablation.startswith("ABL-07"):
        drop = {
            "aa_involves_G",
            "aa_involves_P",
            "aa_involves_C",
            "aa_involves_R",
            "aa_to_special",
            "aa_from_special",
            "aa_class_changed",
            "blosum62_approx",
            "aa_change_te",
        }
        return [c for c in cols if c not in drop]
    if ablation.startswith("ABL-08"):
        drop = {"cat1_multipop", "cat1_AFR", "cat1_NFE", "cat1_gnomADe", "cat1_gnomADg", "cat1_te"}
        return [c for c in cols if c not in drop]
    return cols


def run_ablation_table(
    prepared: PreparedData,
    lgbm_params: dict[str, object] | None,
    out_dir: Path,
    n_estimators: int = 600,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    row_records = []
    prediction_records = []
    params = dict(lgbm_params or {})
    params["n_estimators"] = min(int(params.get("n_estimators", n_estimators)), n_estimators)

    for fold in folds:
        engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
        train = engineer.fit_transform(prepared.master.iloc[fold.train_idx].copy())
        val = engineer.transform(prepared.master.iloc[fold.val_idx].copy())
        y_train = train["Label"]
        y_val = val["Label"]
        acmg_score = acmg_rule_probability(val)

        for ablation in ABLATION_DEFINITIONS:
            if ablation.startswith("ABL-09") or ablation.startswith("ABL-10"):
                continue
            selected = feature_subset_columns(train, ablation)
            x_train = train[selected]
            x_val = val.reindex(columns=selected)
            if ablation.startswith("ABL-04"):
                x_train = x_train.fillna(0.0)
                x_val = x_val.fillna(0.0)
            model = make_lgbm(params)
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="auc")
            score = model.predict_proba(x_val)[:, 1]
            f1_threshold, _ = best_f1_macro_threshold(y_val, score)
            for threshold_name, threshold in [("default_0.5", 0.5), ("f1_macro_opt", f1_threshold)]:
                pred = (score >= threshold).astype(int)
                row_records.append(
                    {
                        "ablation": ablation,
                        "description": ABLATION_DEFINITIONS[ablation],
                        "fold": fold.fold,
                        "threshold_name": threshold_name,
                        "threshold": threshold,
                        "CV_AUC": roc_auc_score(y_val, score),
                        "CV_F1macro": f1_score(y_val, pred, average="macro"),
                    }
                )
            if ablation == "ABL-05_all_with_miss_flags":
                for idx, variant_id, label, prob, acmg_prob in zip(
                    fold.val_idx,
                    val["Variant_ID"],
                    y_val,
                    score,
                    acmg_score,
                ):
                    prediction_records.append(
                        {
                            "fold": fold.fold,
                            "row_index": idx,
                            "Variant_ID": variant_id,
                            "Label": label,
                            "lightgbm_probability": prob,
                            "acmg_probability": acmg_prob,
                        }
                    )

        selected = feature_subset_columns(train, "ABL-05_all_with_miss_flags")
        x_train = train[selected]
        x_val = val.reindex(columns=selected)
        et = make_extra_trees()
        et.fit(x_train.fillna(-999), y_train)
        score = et.predict_proba(x_val.fillna(-999))[:, 1]
        pred = score >= 0.5
        row_records.append(
            {
                "ablation": "ABL-09_single_LGBM_vs_extra_trees",
                "description": ABLATION_DEFINITIONS["ABL-09_single_LGBM_vs_extra_trees"],
                "fold": fold.fold,
                "threshold_name": "extra_trees_default_0.5",
                "threshold": 0.5,
                "CV_AUC": roc_auc_score(y_val, score),
                "CV_F1macro": f1_score(y_val, pred, average="macro"),
            }
        )

    rows = pd.DataFrame(row_records)
    full = rows[(rows["ablation"] == "ABL-05_all_with_miss_flags") & (rows["threshold_name"] == "f1_macro_opt")]
    full_auc = float(full["CV_AUC"].mean())
    full_f1 = float(full["CV_F1macro"].mean())
    summary = (
        rows.groupby(["ablation", "description", "threshold_name"])[["CV_AUC", "CV_F1macro"]]
        .mean()
        .reset_index()
    )
    summary["vs_Full_delta_AUC"] = summary["CV_AUC"] - full_auc
    summary["vs_Full_delta_F1"] = summary["CV_F1macro"] - full_f1
    summary["Conclusion"] = np.where(
        summary["ablation"] == "ABL-05_all_with_miss_flags",
        "Full proposed feature set reference.",
        "Compare delta against full proposed model.",
    )
    default = summary[
        (summary["ablation"] == "ABL-05_all_with_miss_flags")
        & (summary["threshold_name"] == "default_0.5")
    ]
    opt = summary[
        (summary["ablation"] == "ABL-05_all_with_miss_flags")
        & (summary["threshold_name"] == "f1_macro_opt")
    ]
    if not default.empty and not opt.empty:
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "ablation": "ABL-10_default_vs_optimized_threshold",
                            "description": ABLATION_DEFINITIONS["ABL-10_default_vs_optimized_threshold"],
                            "threshold_name": "comparison",
                            "CV_AUC": float(opt["CV_AUC"].iloc[0]),
                            "CV_F1macro": float(opt["CV_F1macro"].iloc[0]),
                            "vs_Full_delta_AUC": 0.0,
                            "vs_Full_delta_F1": float(opt["CV_F1macro"].iloc[0] - default["CV_F1macro"].iloc[0]),
                            "Conclusion": "F1 threshold optimization effect on full model.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    rows.to_csv(out_dir / "ablation_fold_results.csv", index=False)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    pd.DataFrame(prediction_records).to_csv(out_dir / "lightgbm_oof_predictions.csv", index=False)
    return summary, pd.DataFrame(prediction_records)


def final_panel_predictions(
    prepared: PreparedData,
    lgbm_params: dict[str, object],
    model_dir: Path,
    out_dir: Path,
    threshold: float,
) -> pd.DataFrame:
    engineer, model, columns = fit_final_lgbm(prepared, lgbm_params, model_dir)
    rows = []
    for dataset, raw_df in {
        "KANSER_unique": prepared.kanser_unique,
        "PAH_unique": prepared.pah_unique,
        "CFTR_unique": prepared.cftr_unique,
    }.items():
        df = engineer.transform(raw_df)
        prob = model.predict_proba(df.reindex(columns=columns))[:, 1]
        pred = (prob >= threshold).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "Variant_ID": df["Variant_ID"],
                    "Label": df["Label"],
                    "predicted_probability": prob,
                    "predicted_label": pred,
                    "threshold": threshold,
                }
            )
        )
    final = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_dir / "final_panel_unique_predictions.csv", index=False)
    return final


def panel_bootstrap_reports(predictions: pd.DataFrame, out_dir: Path, n_bootstrap: int = 1000) -> pd.DataFrame:
    rows = []
    for dataset, group in predictions.groupby("dataset"):
        ci = bootstrap_metric_ci(
            group["Label"].to_numpy(),
            group["predicted_probability"].to_numpy(),
            float(group["threshold"].iloc[0]),
            n_bootstrap=n_bootstrap,
        )
        ci.insert(0, "dataset", dataset)
        rows.append(ci)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(out_dir / "panel_unique_bootstrap_ci.csv", index=False)
    return result


def write_master_prompt_report(
    checklist: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    stats_tests: pd.DataFrame,
    calibration: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# TEKNOFEST Master Prompt Implementation Report

## Critical Checklist

{checklist.to_markdown(index=False)}

## Ablation Summary

{ablation_summary.to_markdown(index=False)}

## Statistical Tests

{stats_tests.to_markdown(index=False)}

## Calibration

{calibration.to_markdown(index=False)}
"""
    out_path.write_text(text, encoding="utf-8")
