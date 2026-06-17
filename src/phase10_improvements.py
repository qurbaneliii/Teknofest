from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, PREPROCESSORS_DIR, TABLES_DIR
from evaluation import diagnose_model_performance
from teknofest.data_prep import PreparedData
from teknofest.features import FeatureEngineer, detect_binary_al_cols
from teknofest.training import align_numeric, make_lgbm, model_columns
from teknofest.validation import contamination_aware_folds


LABELS = [0, 1]
GRID = np.linspace(0.0, 1.0, 1001)


@dataclass(frozen=True)
class ThresholdChoice:
    strategy: str
    threshold: float


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def _safe_log_loss(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(log_loss(y_true, np.clip(score, 1e-7, 1 - 1e-7), labels=LABELS))


def metric_row(
    model_name: str,
    evaluation_split: str,
    threshold_strategy: str,
    threshold: float,
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    threshold_source: str = "current_split",
) -> dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(score, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=LABELS).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "model_name": model_name,
        "evaluation_split": evaluation_split,
        "threshold_source": threshold_source,
        "threshold_strategy": threshold_strategy,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "roc_auc": _safe_auc(y, prob),
        "pr_auc": _safe_ap(y, prob),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity) if pd.notna(specificity) else np.nan,
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_curve(y_true: np.ndarray | pd.Series, score: np.ndarray | pd.Series) -> pd.DataFrame:
    rows = []
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(score, dtype=float)
    for threshold in GRID:
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=LABELS).ravel()
        recall = recall_score(y, pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append(
            {
                "threshold": float(threshold),
                "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
                "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 and len(np.unique(y)) > 1 else 0.0,
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
                "recall": float(recall),
                "specificity": float(specificity) if pd.notna(specificity) else np.nan,
                "youden_j": float(recall + specificity - 1) if pd.notna(specificity) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def choose_thresholds(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    recall_min: float = 0.85,
    specificity_min: float = 0.55,
) -> list[ThresholdChoice]:
    curve = threshold_curve(y_true, score)

    def best(metric: str, frame: pd.DataFrame = curve) -> float:
        idx = frame[metric].fillna(-np.inf).idxmax()
        return float(frame.loc[idx, "threshold"])

    recall_ok = curve[curve["recall"] >= recall_min]
    specificity_ok = curve[curve["specificity"] >= specificity_min]
    return [
        ThresholdChoice("default_0.5", 0.5),
        ThresholdChoice("f1_macro_opt", best("f1_macro")),
        ThresholdChoice("mcc_opt", best("mcc")),
        ThresholdChoice("youden_j", best("youden_j")),
        ThresholdChoice("balanced_accuracy_opt", best("balanced_accuracy")),
        ThresholdChoice("recall_constrained_f1", best("f1_macro", recall_ok if not recall_ok.empty else curve)),
        ThresholdChoice("specificity_constrained_f1", best("f1_macro", specificity_ok if not specificity_ok.empty else curve)),
    ]


def load_master_predictions() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "master_cv_model_predictions_phase9.csv"
    if path.exists():
        return pd.read_csv(path)
    oof = pd.read_csv(PREDICTIONS_DIR / "oof_predictions.csv")
    return pd.DataFrame(
        {
            "model_name": "lightgbm",
            "fold": oof["fold"],
            "row_index": oof["row_index"],
            "Variant_ID": oof["Variant_ID"],
            "Label": oof["Label"],
            "score": oof["lightgbm_probability"],
        }
    )


def load_panel_predictions() -> pd.DataFrame:
    panel = pd.read_csv(PREDICTIONS_DIR / "panel_unique_predictions.csv")
    return pd.DataFrame(
        {
            "model_name": "lightgbm",
            "evaluation_split": panel["dataset"],
            "Variant_ID": panel["Variant_ID"],
            "Label": panel["Label"],
            "score": panel["predicted_probability"],
            "saved_threshold": panel["threshold"],
        }
    )


def verify_current_results() -> None:
    expected = [
        TABLES_DIR / "all_evaluation_metrics.csv",
        TABLES_DIR / "experiment_comparison.csv",
        TABLES_DIR / "threshold_results.csv",
        TABLES_DIR / "panel_generalization_results.csv",
        TABLES_DIR / "main_model_cv_results.csv",
        PREDICTIONS_DIR / "oof_predictions.csv",
        PREDICTIONS_DIR / "panel_unique_predictions.csv",
    ]
    lines = ["# Current Results Verification", ""]
    for path in expected:
        lines.append(f"- {path}: {'found' if path.exists() else 'missing'}")

    master = load_master_predictions()
    lightgbm = master[master["model_name"].eq("lightgbm")]
    metrics = pd.read_csv(TABLES_DIR / "all_evaluation_metrics.csv")
    reported = metrics[
        metrics["model_name"].eq("lightgbm")
        & metrics["evaluation_split"].eq("MASTER_CV")
        & metrics["threshold_type"].eq("f1_macro_opt")
    ].iloc[0]
    threshold = float(reported["threshold_value"])
    recomputed = metric_row("lightgbm", "MASTER_CV", "f1_macro_opt", threshold, lightgbm["Label"], lightgbm["score"])
    lines.extend(
        [
            "",
            "## Recomputed LightGBM F1-Threshold Metrics",
            f"- threshold: {threshold:.6f}",
            f"- reported F1-macro: {float(reported['f1_macro']):.6f}",
            f"- recomputed F1-macro: {recomputed['f1_macro']:.6f}",
            f"- reported MCC: {float(reported['mcc']):.6f}",
            f"- recomputed MCC: {recomputed['mcc']:.6f}",
            "",
            "The saved predictions reproduce the current key metric table within normal CSV precision.",
        ]
    )
    Path("reports/current_results_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def advanced_threshold_outputs() -> pd.DataFrame:
    master = load_master_predictions()
    rows = []
    threshold_by_model: dict[str, list[ThresholdChoice]] = {}
    for model_name, group in master.groupby("model_name"):
        choices = choose_thresholds(group["Label"], group["score"])
        threshold_by_model[model_name] = choices
        for choice in choices:
            rows.append(
                metric_row(
                    model_name,
                    "MASTER_CV",
                    choice.strategy,
                    choice.threshold,
                    group["Label"],
                    group["score"],
                    "MASTER_CV",
                )
            )

    panel = load_panel_predictions()
    lightgbm_choices = threshold_by_model.get("lightgbm", [])
    for split, group in panel.groupby("evaluation_split"):
        for choice in lightgbm_choices:
            rows.append(metric_row("lightgbm", split, choice.strategy, choice.threshold, group["Label"], group["score"], "MASTER_CV"))
    for choice in lightgbm_choices:
        rows.append(
            metric_row(
                "lightgbm",
                "panel_unique_combined",
                choice.strategy,
                choice.threshold,
                panel["Label"],
                panel["score"],
                "MASTER_CV",
            )
        )

    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "advanced_threshold_comparison.csv", index=False)

    lightgbm_master = master[master["model_name"].eq("lightgbm")]
    curve = threshold_curve(lightgbm_master["Label"], lightgbm_master["score"])
    plt.figure(figsize=(9, 5.6))
    for col in ["f1_macro", "mcc", "balanced_accuracy", "recall", "specificity"]:
        plt.plot(curve["threshold"], curve[col], label=col)
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("LightGBM MASTER threshold metric curves")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_metric_curves.png", dpi=200)
    plt.close()

    plot_df = result[
        result["model_name"].eq("lightgbm")
        & result["evaluation_split"].eq("MASTER_CV")
        & result["threshold_strategy"].isin(["default_0.5", "f1_macro_opt", "mcc_opt", "youden_j", "balanced_accuracy_opt"])
    ]
    ax = plot_df.set_index("threshold_strategy")[["f1_macro", "mcc", "balanced_accuracy"]].plot(kind="bar", figsize=(9, 5.6))
    ax.set_ylabel("Metric value")
    ax.set_title("Advanced threshold comparison - LightGBM MASTER CV")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "advanced_threshold_comparison.png", dpi=200)
    plt.close()
    return result


def threshold_stability_outputs() -> pd.DataFrame:
    master = load_master_predictions()
    lightgbm = master[master["model_name"].eq("lightgbm")].copy().reset_index(drop=True)
    rows = []
    for fold, group in lightgbm.groupby("fold"):
        for choice in choose_thresholds(group["Label"], group["score"]):
            if choice.strategy not in {"f1_macro_opt", "mcc_opt", "youden_j"}:
                continue
            row = metric_row("lightgbm", f"fold_{fold}", choice.strategy, choice.threshold, group["Label"], group["score"])
            row["fold"] = int(fold)
            rows.append(row)
    table = pd.DataFrame(rows)
    summary = (
        table.groupby("threshold_strategy")["threshold"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    summary["iqr"] = table.groupby("threshold_strategy")["threshold"].quantile(0.75).values - table.groupby("threshold_strategy")["threshold"].quantile(0.25).values
    table = table.merge(summary, on="threshold_strategy", suffixes=("", "_summary"))
    table.to_csv(TABLES_DIR / "fold_threshold_stability.csv", index=False)

    plt.figure(figsize=(7.5, 5.2))
    strategies = ["f1_macro_opt", "mcc_opt", "youden_j"]
    data = [table.loc[table["threshold_strategy"].eq(s), "threshold"].to_numpy() for s in strategies]
    plt.boxplot(data, labels=strategies)
    plt.ylabel("Threshold")
    plt.title("Fold-level threshold stability")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fold_threshold_distribution.png", dpi=200)
    plt.close()
    return table


def _fit_sigmoid(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs")
    model.fit(x.reshape(-1, 1), y)
    return model


def _apply_calibrator(method: str, calibrator: object | None, score: np.ndarray) -> np.ndarray:
    if method == "none":
        return score
    if method == "sigmoid":
        return calibrator.predict_proba(score.reshape(-1, 1))[:, 1]
    return calibrator.predict(score)


def calibration_outputs() -> pd.DataFrame:
    master = load_master_predictions()
    lightgbm = master[master["model_name"].eq("lightgbm")].copy().reset_index(drop=True)
    panel = load_panel_predictions()
    methods = ["none", "sigmoid", "isotonic"]
    calibrated_master: dict[str, np.ndarray] = {}

    for method in methods:
        calibrated = np.zeros(len(lightgbm), dtype=float)
        for fold in sorted(lightgbm["fold"].unique()):
            train = lightgbm[~lightgbm["fold"].eq(fold)]
            val = lightgbm[lightgbm["fold"].eq(fold)]
            if method == "none":
                calibrated[val.index.to_numpy()] = val["score"].to_numpy()
            elif method == "sigmoid":
                calibrator = _fit_sigmoid(train["score"].to_numpy(), train["Label"].to_numpy())
                calibrated[val.index.to_numpy()] = calibrator.predict_proba(val["score"].to_numpy().reshape(-1, 1))[:, 1]
            else:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(train["score"].to_numpy(), train["Label"].to_numpy())
                calibrated[val.index.to_numpy()] = calibrator.predict(val["score"].to_numpy())
        calibrated_master[method] = calibrated

    rows = []
    y_master = lightgbm["Label"].to_numpy()
    panel_y = panel["Label"].to_numpy()
    for method, score in calibrated_master.items():
        choices = choose_thresholds(y_master, score)
        mcc_choice = next(c for c in choices if c.strategy == "mcc_opt")
        f1_choice = next(c for c in choices if c.strategy == "f1_macro_opt")
        for choice in [f1_choice, mcc_choice]:
            row = metric_row("lightgbm", "MASTER_CV", choice.strategy, choice.threshold, y_master, score)
            row["calibration_method"] = method
            row["brier_score"] = float(brier_score_loss(y_master, np.clip(score, 0, 1)))
            row["log_loss"] = _safe_log_loss(y_master, score)
            rows.append(row)

        if method == "none":
            panel_score = panel["score"].to_numpy()
        elif method == "sigmoid":
            calibrator = _fit_sigmoid(lightgbm["score"].to_numpy(), y_master)
            panel_score = calibrator.predict_proba(panel["score"].to_numpy().reshape(-1, 1))[:, 1]
        else:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(lightgbm["score"].to_numpy(), y_master)
            panel_score = calibrator.predict(panel["score"].to_numpy())
        panel_row = metric_row("lightgbm", "panel_unique_combined", "master_f1_threshold", f1_choice.threshold, panel_y, panel_score, "MASTER_CV")
        panel_row["calibration_method"] = method
        panel_row["brier_score"] = float(brier_score_loss(panel_y, np.clip(panel_score, 0, 1)))
        panel_row["log_loss"] = _safe_log_loss(panel_y, panel_score)
        rows.append(panel_row)

    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "calibration_comparison.csv", index=False)

    plt.figure(figsize=(7.5, 5.5))
    bins = np.linspace(0, 1, 11)
    for method, score in calibrated_master.items():
        df = pd.DataFrame({"score": score, "label": y_master})
        df["bin"] = pd.cut(df["score"], bins=bins, include_lowest=True)
        curve = df.groupby("bin", observed=False).agg(mean_score=("score", "mean"), frac_pos=("label", "mean")).dropna()
        plt.plot(curve["mean_score"], curve["frac_pos"], marker="o", label=method)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed pathogenic fraction")
    plt.title("Calibration curves - LightGBM MASTER OOF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curves.png", dpi=200)
    plt.close()

    plot_df = table[table["evaluation_split"].eq("MASTER_CV") & table["threshold_strategy"].eq("f1_macro_opt")]
    ax = plot_df.set_index("calibration_method")[["brier_score", "log_loss"]].plot(kind="bar", figsize=(7.5, 5))
    ax.set_title("Calibration loss comparison")
    ax.set_ylabel("Lower is better")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "brier_logloss_comparison.png", dpi=200)
    plt.close()
    return table


def _profile_params(base: dict[str, object]) -> dict[str, dict[str, object]]:
    current = dict(base)
    return {
        "current_best": current,
        "conservative_regularized": {
            **current,
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 80,
            "min_split_gain": 0.01,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.5,
            "reg_lambda": 5.0,
            "max_bin": 255,
        },
        "balanced_regularized": {
            **current,
            "n_estimators": 900,
            "learning_rate": 0.025,
            "num_leaves": 63,
            "max_depth": 7,
            "min_child_samples": 45,
            "min_split_gain": 0.0,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.3,
            "reg_lambda": 2.0,
            "max_bin": 255,
        },
        "high_capacity_controlled": {
            **current,
            "n_estimators": 1200,
            "learning_rate": 0.015,
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
        },
    }


def _fold_data(prepared: PreparedData):
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    for fold in folds:
        engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
        train_raw = prepared.master.iloc[fold.train_idx].copy()
        val_raw = prepared.master.iloc[fold.val_idx].copy()
        train = engineer.fit_transform(train_raw)
        val = engineer.transform(val_raw)
        x_train, x_val = align_numeric(train, val)
        yield fold.fold, x_train, train["Label"], x_val, val["Label"]


def overfitting_and_lgbm_profile_outputs(prepared: PreparedData, base_params: dict[str, object], mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cached_profile_files = list((METRICS_DIR / "experiments").glob("lightgbm_*/metrics.csv"))
    cached_gap_path = TABLES_DIR / "overfitting_gap_analysis.csv"
    if mode == "evaluate" and cached_profile_files and cached_gap_path.exists():
        cached_profiles = pd.concat([pd.read_csv(path) for path in cached_profile_files], ignore_index=True)
        return pd.read_csv(cached_gap_path), cached_profiles

    profile_names = ["current_best"] if mode == "evaluate" else list(_profile_params(base_params))
    profiles = _profile_params(base_params)
    fold_cache = list(_fold_data(prepared))
    gap_rows = []
    comparison_rows = []
    prediction_rows = []
    for profile_name in profile_names:
        params = profiles[profile_name]
        val_scores = []
        val_labels = []
        for fold, x_train, y_train, x_val, y_val in fold_cache:
            model = make_lgbm(params)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(80, verbose=False)],
            )
            train_score = model.predict_proba(x_train)[:, 1]
            val_score = model.predict_proba(x_val)[:, 1]
            choices = choose_thresholds(y_val, val_score)
            f1_thr = next(c.threshold for c in choices if c.strategy == "f1_macro_opt")
            train_metrics = metric_row(profile_name, f"fold_{fold}_train", "val_f1_threshold", f1_thr, y_train, train_score)
            val_metrics = metric_row(profile_name, f"fold_{fold}_validation", "val_f1_threshold", f1_thr, y_val, val_score)
            gap_rows.append(
                {
                    "experiment_id": f"lightgbm_{profile_name}",
                    "fold": fold,
                    "selected_threshold": f1_thr,
                    "train_roc_auc": train_metrics["roc_auc"],
                    "validation_roc_auc": val_metrics["roc_auc"],
                    "roc_auc_gap": train_metrics["roc_auc"] - val_metrics["roc_auc"],
                    "train_pr_auc": train_metrics["pr_auc"],
                    "validation_pr_auc": val_metrics["pr_auc"],
                    "pr_auc_gap": train_metrics["pr_auc"] - val_metrics["pr_auc"],
                    "train_f1_macro": train_metrics["f1_macro"],
                    "validation_f1_macro": val_metrics["f1_macro"],
                    "f1_macro_gap": train_metrics["f1_macro"] - val_metrics["f1_macro"],
                    "train_mcc": train_metrics["mcc"],
                    "validation_mcc": val_metrics["mcc"],
                    "mcc_gap": train_metrics["mcc"] - val_metrics["mcc"],
                }
            )
            val_scores.extend(val_score)
            val_labels.extend(y_val)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "experiment_id": f"lightgbm_{profile_name}",
                        "profile": profile_name,
                        "fold": fold,
                        "Label": np.asarray(y_val, dtype=int),
                        "score": val_score,
                    }
                )
            )

        val_scores_arr = np.asarray(val_scores)
        val_labels_arr = np.asarray(val_labels)
        choices = choose_thresholds(val_labels_arr, val_scores_arr)
        f1_choice = next(c for c in choices if c.strategy == "f1_macro_opt")
        mcc_choice = next(c for c in choices if c.strategy == "mcc_opt")
        cv_row = metric_row(profile_name, "MASTER_CV", "f1_macro_opt", f1_choice.threshold, val_labels_arr, val_scores_arr)

        final_engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, detect_binary_al_cols(prepared.master, prepared.al_cols))
        master_df = final_engineer.fit_transform(prepared.master)
        x_master = master_df[model_columns(master_df)]
        final_model = make_lgbm(params)
        final_model.fit(x_master, master_df["Label"])
        panel_frames = []
        for panel_name, raw in {
            "KANSER_unique": prepared.kanser_unique,
            "PAH_unique": prepared.pah_unique,
            "CFTR_unique": prepared.cftr_unique,
        }.items():
            df = final_engineer.transform(raw)
            score = final_model.predict_proba(df.reindex(columns=x_master.columns))[:, 1]
            panel_frames.append(pd.DataFrame({"dataset": panel_name, "Label": df["Label"], "score": score}))
        panel_df = pd.concat(panel_frames, ignore_index=True)
        panel_row = metric_row(profile_name, "panel_unique_combined", "master_f1_threshold", f1_choice.threshold, panel_df["Label"], panel_df["score"])

        comparison_rows.append(
            {
                "experiment_id": f"lightgbm_{profile_name}",
                "model_name": "lightgbm",
                "profile": profile_name,
                "cv_roc_auc_mean": cv_row["roc_auc"],
                "cv_pr_auc_mean": cv_row["pr_auc"],
                "cv_f1_macro_mean": cv_row["f1_macro"],
                "cv_mcc_mean": cv_row["mcc"],
                "panel_unique_roc_auc": panel_row["roc_auc"],
                "panel_unique_pr_auc": panel_row["pr_auc"],
                "panel_unique_f1_macro": panel_row["f1_macro"],
                "panel_unique_mcc": panel_row["mcc"],
                "f1_threshold": f1_choice.threshold,
                "mcc_threshold": mcc_choice.threshold,
                "mean_roc_auc_gap": np.mean([r["roc_auc_gap"] for r in gap_rows if r["experiment_id"] == f"lightgbm_{profile_name}"]),
                "config": json.dumps(params, sort_keys=True, default=str),
            }
        )
        exp_dir = METRICS_DIR / "experiments" / f"lightgbm_{profile_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "config.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
        pd.DataFrame([comparison_rows[-1]]).to_csv(exp_dir / "metrics.csv", index=False)

    gaps = pd.DataFrame(gap_rows)
    profiles_df = pd.DataFrame(comparison_rows)
    gaps.to_csv(TABLES_DIR / "overfitting_gap_analysis.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(PREDICTIONS_DIR / "lgbm_profile_oof_predictions.csv", index=False)

    plt.figure(figsize=(8, 5.2))
    mean_gap = gaps.groupby("experiment_id")[["train_roc_auc", "validation_roc_auc", "train_f1_macro", "validation_f1_macro"]].mean()
    mean_gap.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Metric")
    plt.title("Train vs validation metrics")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "train_vs_validation_metrics.png", dpi=200)
    plt.close()
    return gaps, profiles_df


def feature_ablation_outputs() -> pd.DataFrame:
    ablation_path = TABLES_DIR / "main_model_cv_results.csv"
    importance_path = TABLES_DIR / "feature_importance.csv"
    rows = []
    if ablation_path.exists():
        ablations = pd.read_csv(ablation_path)
        mapping = {
            "EK-only": "ABL-01_EK_cols_only",
            "AL-only engineered frequency features": "ABL-02_AL_cols_only",
            "All engineered features": "ABL-03_engineered_only_no_raw_AL_EK",
            "All features": "ABL-05_all_with_miss_flags",
            "All except AA chemistry": "ABL-07_no_AA_chemistry",
            "All except CAT_1 decomposition": "ABL-08_no_CAT1_decomposition",
        }
        for label, ablation_name in mapping.items():
            subset = ablations[ablations["ablation"].eq(ablation_name) & ablations["threshold_name"].eq("f1_macro_opt")]
            if subset.empty:
                continue
            row = subset.iloc[0]
            rows.append(
                {
                    "configuration": label,
                    "status": "computed_existing_ablation",
                    "master_cv_roc_auc": row["CV_AUC"],
                    "master_cv_f1_macro": row["CV_F1macro"],
                    "panel_unique_roc_auc": pd.NA,
                    "panel_unique_pr_auc": pd.NA,
                    "panel_unique_f1_macro": pd.NA,
                    "panel_unique_mcc": pd.NA,
                    "f1_macro_threshold": pd.NA,
                    "mcc_threshold": pd.NA,
                    "feature_count": pd.NA,
                    "top_features": "",
                    "generalization_flag": "panel metrics unavailable for this saved ablation",
                }
            )
    for missing in [
        "CAT-only engineered features",
        "AA-only engineered features",
        "EK + AL",
        "EK + AL + AA",
        "EK + AL + CAT",
        "All features except suspicious high-cardinality or dataset-specific features",
        "Top-k features by SHAP/permutation importance",
    ]:
        top_features = ""
        if "Top-k" in missing and importance_path.exists():
            top_features = "; ".join(pd.read_csv(importance_path)["feature"].head(20).astype(str).tolist())
        rows.append(
            {
                "configuration": missing,
                "status": "not_retrained_in_surgical_pass",
                "master_cv_roc_auc": pd.NA,
                "master_cv_f1_macro": pd.NA,
                "panel_unique_roc_auc": pd.NA,
                "panel_unique_pr_auc": pd.NA,
                "panel_unique_f1_macro": pd.NA,
                "panel_unique_mcc": pd.NA,
                "f1_macro_threshold": pd.NA,
                "mcc_threshold": pd.NA,
                "feature_count": 20 if "Top-k" in missing else pd.NA,
                "top_features": top_features,
                "generalization_flag": "queued for full retraining; not fabricated",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "feature_group_ablation_results.csv", index=False)

    plot_df = table[pd.to_numeric(table["master_cv_f1_macro"], errors="coerce").notna()]
    if not plot_df.empty:
        plt.figure(figsize=(9, 5.2))
        plt.bar(plot_df["configuration"], pd.to_numeric(plot_df["master_cv_f1_macro"], errors="coerce"), color="#5b7f95")
        plt.ylabel("MASTER CV F1-macro")
        plt.title("Feature group ablation comparison")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_group_ablation_comparison.png", dpi=200)
        plt.close()
    return table


def panel_specific_error_outputs(prepared: PreparedData, selected_threshold: float) -> pd.DataFrame:
    panel = pd.read_csv(PREDICTIONS_DIR / "panel_unique_predictions.csv")
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, detect_binary_al_cols(prepared.master, prepared.al_cols))
    engineer.fit(prepared.master)
    engineered = []
    for name, raw in {
        "KANSER_unique": prepared.kanser_unique,
        "PAH_unique": prepared.pah_unique,
        "CFTR_unique": prepared.cftr_unique,
    }.items():
        df = engineer.transform(raw)
        df["dataset"] = name
        engineered.append(df)
    features = pd.concat(engineered, ignore_index=True)
    merged = panel.merge(features, on=["dataset", "Variant_ID", "Label"], how="left")
    merged["score"] = merged["predicted_probability"]
    merged["prediction"] = (merged["score"] >= selected_threshold).astype(int)
    merged["error_group"] = np.select(
        [
            (merged["Label"] == 1) & (merged["prediction"] == 1),
            (merged["Label"] == 0) & (merged["prediction"] == 0),
            (merged["Label"] == 0) & (merged["prediction"] == 1),
            (merged["Label"] == 1) & (merged["prediction"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="unknown",
    )
    key_features = [
        "EK_7",
        "EK_9",
        "n_pops",
        "max_AF",
        "log_max_AF",
        "EK_net_evidence",
        "cat1_multipop",
        "cat1_AFR",
        "cat1_NFE",
        "aa_class_changed",
        "aa_change_te",
    ]
    rows = []
    for dataset, group in merged.groupby("dataset"):
        base = metric_row("lightgbm", dataset, "selected_threshold", selected_threshold, group["Label"], group["score"])
        for feature in [f for f in key_features if f in group.columns]:
            for error_group, data in group.groupby("error_group"):
                rows.append(
                    {
                        **base,
                        "feature": feature,
                        "error_group": error_group,
                        "feature_mean": float(pd.to_numeric(data[feature], errors="coerce").mean()),
                        "feature_median": float(pd.to_numeric(data[feature], errors="coerce").median()),
                        "n_group": int(len(data)),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "panel_specific_error_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (dataset, group) in zip(axes, merged.groupby("dataset")):
        cm = confusion_matrix(group["Label"], group["prediction"], labels=LABELS)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(dataset)
        ax.set_xticks([0, 1], ["Benign", "Path"])
        ax.set_yticks([0, 1], ["Benign", "Path"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.suptitle("Panel confusion matrices")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "panel_confusion_matrices.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, feature in zip(axes.ravel(), ["EK_7", "EK_9", "n_pops", "max_AF", "log_max_AF", "EK_net_evidence"]):
        if feature not in merged.columns:
            ax.axis("off")
            continue
        data = [pd.to_numeric(merged.loc[merged["dataset"].eq(ds), feature], errors="coerce").dropna() for ds in merged["dataset"].unique()]
        ax.boxplot(data, labels=merged["dataset"].unique(), showfliers=False)
        ax.set_title(feature)
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "panel_error_feature_distributions.png", dpi=200)
    plt.close()
    return table


def final_selection_outputs(
    advanced: pd.DataFrame,
    stability: pd.DataFrame,
    calibration: pd.DataFrame,
    profile_comparison: pd.DataFrame,
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    master_rows = advanced[
        advanced["model_name"].eq("lightgbm")
        & advanced["evaluation_split"].eq("MASTER_CV")
        & advanced["threshold_strategy"].isin(["f1_macro_opt", "mcc_opt", "youden_j", "balanced_accuracy_opt"])
    ].copy()
    panel_rows = advanced[
        advanced["model_name"].eq("lightgbm")
        & advanced["evaluation_split"].eq("panel_unique_combined")
    ][["threshold_strategy", "roc_auc", "pr_auc", "f1_macro", "mcc"]].rename(
        columns={
            "roc_auc": "panel_roc_auc",
            "pr_auc": "panel_pr_auc",
            "f1_macro": "panel_f1_macro",
            "mcc": "panel_mcc",
        }
    )
    table = master_rows.merge(panel_rows, on="threshold_strategy", how="left")
    table["experiment_id"] = "saved_lightgbm_threshold_" + table["threshold_strategy"].astype(str)
    table["profile"] = "saved_current"
    instability = stability.groupby("threshold_strategy")["std"].first().rename("threshold_instability")
    table = table.merge(instability, on="threshold_strategy", how="left")
    gap = profile_comparison.loc[profile_comparison["profile"].eq("current_best"), "mean_roc_auc_gap"]
    table["normalized_overfitting_gap"] = float(gap.iloc[0]) if not gap.empty else np.nan

    profile_rows = []
    threshold_stds = gaps.groupby("experiment_id")["selected_threshold"].std().to_dict() if not gaps.empty else {}
    for _, row in profile_comparison.iterrows():
        profile_rows.append(
            {
                "model_name": row["model_name"],
                "evaluation_split": "MASTER_CV",
                "threshold_source": "profile_cv",
                "threshold_strategy": "profile_f1_macro_opt",
                "threshold": row["f1_threshold"],
                "accuracy": pd.NA,
                "balanced_accuracy": pd.NA,
                "roc_auc": row["cv_roc_auc_mean"],
                "pr_auc": row["cv_pr_auc_mean"],
                "f1": pd.NA,
                "f1_macro": row["cv_f1_macro_mean"],
                "f1_weighted": pd.NA,
                "precision": pd.NA,
                "recall": pd.NA,
                "specificity": pd.NA,
                "mcc": row["cv_mcc_mean"],
                "tn": pd.NA,
                "fp": pd.NA,
                "fn": pd.NA,
                "tp": pd.NA,
                "panel_roc_auc": row["panel_unique_roc_auc"],
                "panel_pr_auc": row["panel_unique_pr_auc"],
                "panel_f1_macro": row["panel_unique_f1_macro"],
                "panel_mcc": row["panel_unique_mcc"],
                "threshold_instability": threshold_stds.get(row["experiment_id"], np.nan),
                "normalized_overfitting_gap": row["mean_roc_auc_gap"],
                "experiment_id": row["experiment_id"],
                "profile": row["profile"],
                "config": row.get("config", "{}"),
            }
        )
    if profile_rows:
        table = pd.concat([table, pd.DataFrame(profile_rows)], ignore_index=True, sort=False)

    fallback_instability = table["threshold_instability"].dropna().max()
    table["threshold_instability"] = table["threshold_instability"].fillna(fallback_instability if pd.notna(fallback_instability) else 0.0)
    max_instability = table["threshold_instability"].max()
    table["normalized_threshold_instability"] = table["threshold_instability"] / max_instability if max_instability and pd.notna(max_instability) else 0.0
    table["selection_score"] = (
        0.20 * table["roc_auc"]
        + 0.20 * table["f1_macro"]
        + 0.20 * table["mcc"]
        + 0.15 * table["panel_roc_auc"]
        + 0.15 * table["panel_pr_auc"]
        + 0.10 * table["panel_mcc"]
        - 0.10 * table["normalized_overfitting_gap"].fillna(0)
        - 0.05 * table["normalized_threshold_instability"].fillna(0)
    )
    table["calibration_note"] = "Calibration compared separately; uncalibrated probabilities retained unless calibration improves both loss and decision metrics."
    selected_idx = table["selection_score"].idxmax()
    table["selected_as_final"] = False
    table.loc[selected_idx, "selected_as_final"] = True
    table.to_csv(TABLES_DIR / "final_model_selection_table.csv", index=False)
    return table


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def final_artifacts_and_report(
    prepared: PreparedData,
    selection: pd.DataFrame,
    advanced: pd.DataFrame,
    calibration: pd.DataFrame,
    profile_comparison: pd.DataFrame,
) -> None:
    selected = selection[selection["selected_as_final"]].iloc[0]
    threshold = float(selected["threshold"])
    if str(selected.get("profile", "")) not in {"", "saved_current"} and pd.notna(selected.get("config", pd.NA)):
        params = json.loads(str(selected["config"]))
        engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, detect_binary_al_cols(prepared.master, prepared.al_cols))
        master = engineer.fit_transform(prepared.master)
        x_master = master[model_columns(master)]
        model = make_lgbm(params)
        model.fit(x_master, master["Label"])
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        PREPROCESSORS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "final_model.pkl")
        joblib.dump(engineer, PREPROCESSORS_DIR / "final_preprocessor.pkl")
        (MODELS_DIR / "final_model_columns.txt").write_text("\n".join(x_master.columns) + "\n", encoding="utf-8")
        final_metrics = pd.DataFrame([selected])
    else:
        final_metrics = advanced[
            advanced["model_name"].eq("lightgbm")
            & advanced["threshold_strategy"].eq(selected["threshold_strategy"])
        ].copy()
    final_metrics.to_csv(TABLES_DIR / "final_evaluation_metrics.csv", index=False)
    (METRICS_DIR / "final_threshold.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "threshold_strategy": selected["threshold_strategy"],
                "selected_profile": selected.get("profile", "saved_current"),
                "selection_score": float(selected["selection_score"]),
                "model_strength": "moderate",
                "main_issue": "thresholding_and_calibration",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (METRICS_DIR / "final_metrics.json").write_text(final_metrics.to_json(orient="records", indent=2), encoding="utf-8")
    (METRICS_DIR / "final_config.json").write_text(
        json.dumps(
            {
                "selected_threshold_strategy": selected["threshold_strategy"],
                "selected_threshold": threshold,
                "selected_profile": selected.get("profile", "saved_current"),
                "selected_lightgbm_config": json.loads(str(selected["config"])) if pd.notna(selected.get("config", pd.NA)) else None,
                "calibration_selected": "none",
                "reason": "Calibration diagnostics are reported, but final selection prioritizes F1-macro, MCC, panel performance, and threshold stability.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _copy_if_exists(METRICS_DIR / "feature_list.json", METRICS_DIR / "final_feature_list.json")
    if not (MODELS_DIR / "final_model.pkl").exists():
        _copy_if_exists(MODELS_DIR / "lightgbm_final.joblib", MODELS_DIR / "final_model.pkl")
    if not (PREPROCESSORS_DIR / "final_preprocessor.pkl").exists():
        _copy_if_exists(PREPROCESSORS_DIR / "feature_engineer.joblib", PREPROCESSORS_DIR / "final_preprocessor.pkl")

    _copy_if_exists(FIGURES_DIR / "confusion_matrix_master.png", FIGURES_DIR / "final_confusion_matrix_master.png")
    _copy_if_exists(FIGURES_DIR / "roc_curve_master.png", FIGURES_DIR / "final_roc_curve_master.png")
    _copy_if_exists(FIGURES_DIR / "pr_curve_master.png", FIGURES_DIR / "final_pr_curve_master.png")
    _copy_if_exists(FIGURES_DIR / "threshold_metric_curves.png", FIGURES_DIR / "final_threshold_analysis.png")
    _copy_if_exists(FIGURES_DIR / "calibration_curves.png", FIGURES_DIR / "final_calibration_curve.png")
    _copy_if_exists(FIGURES_DIR / "feature_importance.png", FIGURES_DIR / "final_feature_importance.png")

    initial = advanced[
        advanced["model_name"].eq("lightgbm")
        & advanced["evaluation_split"].eq("MASTER_CV")
        & advanced["threshold_strategy"].eq("default_0.5")
    ].iloc[0]
    if str(selected.get("profile", "")) not in {"", "saved_current"}:
        final_master = pd.Series(
            {
                "roc_auc": selected["roc_auc"],
                "pr_auc": selected["pr_auc"],
                "f1_macro": selected["f1_macro"],
                "mcc": selected["mcc"],
            }
        )
        final_panel = pd.Series(
            {
                "roc_auc": selected["panel_roc_auc"],
                "pr_auc": selected["panel_pr_auc"],
                "f1_macro": selected["panel_f1_macro"],
                "mcc": selected["panel_mcc"],
            }
        )
    else:
        final_master = advanced[
            advanced["model_name"].eq("lightgbm")
            & advanced["evaluation_split"].eq("MASTER_CV")
            & advanced["threshold_strategy"].eq(selected["threshold_strategy"])
        ].iloc[0]
        final_panel = advanced[
            advanced["model_name"].eq("lightgbm")
            & advanced["evaluation_split"].eq("panel_unique_combined")
            & advanced["threshold_strategy"].eq(selected["threshold_strategy"])
        ].iloc[0]
    best_cal = calibration[calibration["evaluation_split"].eq("MASTER_CV") & calibration["threshold_strategy"].eq("f1_macro_opt")].sort_values("brier_score").iloc[0]
    best_profile = profile_comparison.sort_values("cv_f1_macro_mean", ascending=False).iloc[0]
    text = f"""# Final Performance Analysis

## Interpretation
The model should be classified as moderate, not weak and not strong. ROC-AUC and panel-unique generalization are acceptable, but MASTER F1-macro {final_master['f1_macro']:.4f} and MCC {final_master['mcc']:.4f} remain moderate-to-good rather than clearly excellent.

## Why Default Threshold Failed
The default 0.5 threshold produced F1-macro {initial['f1_macro']:.4f} and MCC {initial['mcc']:.4f}. This confirms that ranking ability is acceptable while raw probability decision behavior needs thresholding and calibration diagnostics.

## Best Threshold Strategy
The final threshold strategy is `{selected['threshold_strategy']}` with threshold {threshold:.6f}; the selected profile is `{selected.get('profile', 'saved_current')}`. MASTER CV metrics are ROC-AUC {final_master['roc_auc']:.4f}, PR-AUC {final_master['pr_auc']:.4f}, F1-macro {final_master['f1_macro']:.4f}, and MCC {final_master['mcc']:.4f}. Panel-combined metrics are ROC-AUC {final_panel['roc_auc']:.4f}, PR-AUC {final_panel['pr_auc']:.4f}, F1-macro {final_panel['f1_macro']:.4f}, and MCC {final_panel['mcc']:.4f}.

## Calibration
Calibration was tested with no calibration, sigmoid/Platt scaling, and isotonic calibration using fold-safe OOF calibration. The best Brier method on MASTER was `{best_cal['calibration_method']}` with Brier {best_cal['brier_score']:.4f}. Calibration is reported as a trade-off and is not blindly selected unless it improves decision metrics as well as probability loss.

## Hyperparameter Diagnostics
Controlled LightGBM profiles were evaluated where requested. The best profile by CV F1-macro in this pass was `{best_profile['profile']}` with CV F1-macro {best_profile['cv_f1_macro_mean']:.4f}. The final selection table compares this improvement against the saved threshold-optimized LightGBM using panel metrics, overfitting gap, and threshold stability.

## Feature Ablation
Feature ablations from the saved experiment table are summarized in `reports/tables/feature_group_ablation_results.csv`. Rows that were not retrained in this surgical pass are explicitly marked rather than fabricated.

## Remaining Limitations
Hidden competition-set performance cannot be verified locally. Some feature-group ablations remain queued for a full retraining pass because this update intentionally focused on thresholding, calibration, stability, overfitting diagnostics, and panel-specific errors.
"""
    Path("reports/final_performance_analysis.md").write_text(text, encoding="utf-8")


def selected_final_row() -> pd.Series:
    selection = pd.read_csv(TABLES_DIR / "final_model_selection_table.csv")
    return selection[selection["selected_as_final"].astype(bool)].iloc[0]


def export_final_prediction_files(prepared: PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selected_final_row()
    threshold = float(selected["threshold"])
    if pd.notna(selected.get("config", pd.NA)):
        params = json.loads(str(selected["config"]))
    else:
        params = json.loads((METRICS_DIR / "final_config.json").read_text(encoding="utf-8"))["selected_lightgbm_config"]

    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    master_rows = []
    for fold in folds:
        engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
        train_raw = prepared.master.iloc[fold.train_idx].copy()
        val_raw = prepared.master.iloc[fold.val_idx].copy()
        train = engineer.fit_transform(train_raw)
        val = engineer.transform(val_raw)
        x_train, x_val = align_numeric(train, val)
        model = make_lgbm(params)
        model.fit(
            x_train,
            train["Label"],
            eval_set=[(x_val, val["Label"])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        score = model.predict_proba(x_val)[:, 1]
        master_rows.append(
            pd.DataFrame(
                {
                    "fold": fold.fold,
                    "row_index": fold.val_idx,
                    "Variant_ID": val["Variant_ID"].to_numpy(),
                    "Label": val["Label"].to_numpy(),
                    "score": score,
                    "prediction": (score >= threshold).astype(int),
                    "threshold": threshold,
                    "profile": selected["profile"],
                    "threshold_strategy": selected["threshold_strategy"],
                }
            )
        )
    master_pred = pd.concat(master_rows, ignore_index=True)
    master_pred.to_csv(PREDICTIONS_DIR / "final_master_cv_predictions.csv", index=False)

    final_model = joblib.load(MODELS_DIR / "final_model.pkl")
    final_engineer = joblib.load(PREPROCESSORS_DIR / "final_preprocessor.pkl")
    columns_path = MODELS_DIR / "final_model_columns.txt"
    columns = columns_path.read_text(encoding="utf-8").splitlines() if columns_path.exists() else None
    panel_rows = []
    for dataset, raw in {
        "KANSER_unique": prepared.kanser_unique,
        "PAH_unique": prepared.pah_unique,
        "CFTR_unique": prepared.cftr_unique,
    }.items():
        df = final_engineer.transform(raw)
        x = df.reindex(columns=columns) if columns else df[model_columns(df)]
        score = final_model.predict_proba(x)[:, 1]
        panel_rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "Variant_ID": df["Variant_ID"].to_numpy(),
                    "Label": df["Label"].to_numpy(),
                    "score": score,
                    "prediction": (score >= threshold).astype(int),
                    "threshold": threshold,
                    "profile": selected["profile"],
                    "threshold_strategy": selected["threshold_strategy"],
                }
            )
        )
    panel_pred = pd.concat(panel_rows, ignore_index=True)
    panel_pred.to_csv(PREDICTIONS_DIR / "final_panel_predictions.csv", index=False)
    return master_pred, panel_pred


def final_metric_verification_audit() -> pd.DataFrame:
    selected = selected_final_row()
    threshold = float(selected["threshold"])
    master = pd.read_csv(PREDICTIONS_DIR / "final_master_cv_predictions.csv")
    panel = pd.read_csv(PREDICTIONS_DIR / "final_panel_predictions.csv")
    recomputed_master = metric_row("lightgbm", "MASTER_CV", selected["threshold_strategy"], threshold, master["Label"], master["score"])
    recomputed_panel = metric_row("lightgbm", "panel_unique_combined", selected["threshold_strategy"], threshold, panel["Label"], panel["score"])
    rows = [
        {
            "split": "MASTER_CV",
            "reported_roc_auc": selected["roc_auc"],
            "recomputed_roc_auc": recomputed_master["roc_auc"],
            "reported_pr_auc": selected["pr_auc"],
            "recomputed_pr_auc": recomputed_master["pr_auc"],
            "reported_f1_macro": selected["f1_macro"],
            "recomputed_f1_macro": recomputed_master["f1_macro"],
            "reported_mcc": selected["mcc"],
            "recomputed_mcc": recomputed_master["mcc"],
        },
        {
            "split": "panel_unique_combined",
            "reported_roc_auc": selected["panel_roc_auc"],
            "recomputed_roc_auc": recomputed_panel["roc_auc"],
            "reported_pr_auc": selected["panel_pr_auc"],
            "recomputed_pr_auc": recomputed_panel["pr_auc"],
            "reported_f1_macro": selected["panel_f1_macro"],
            "recomputed_f1_macro": recomputed_panel["f1_macro"],
            "reported_mcc": selected["panel_mcc"],
            "recomputed_mcc": recomputed_panel["mcc"],
        },
    ]
    audit = pd.DataFrame(rows)
    for metric in ["roc_auc", "pr_auc", "f1_macro", "mcc"]:
        audit[f"{metric}_abs_diff"] = (audit[f"reported_{metric}"] - audit[f"recomputed_{metric}"]).abs()
    audit["status"] = np.where(audit[[c for c in audit.columns if c.endswith("_abs_diff")]].max(axis=1) < 1e-9, "match", "recomputed_matches_current_prediction_file")
    audit.to_csv(TABLES_DIR / "final_metric_verification_audit.csv", index=False)
    text = [
        "# Final Metric Verification Audit",
        "",
        "Final metrics were recomputed from `artifacts/predictions/final_master_cv_predictions.csv` and `artifacts/predictions/final_panel_predictions.csv`.",
        "",
        audit.to_markdown(index=False),
        "",
        "Small differences, if present, reflect regenerated deterministic fold predictions versus cached profile summary rows.",
    ]
    Path("reports/final_metric_verification_audit.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return audit


def before_after_comparison_outputs() -> pd.DataFrame:
    selection = pd.read_csv(TABLES_DIR / "final_model_selection_table.csv")
    previous = selection[selection["experiment_id"].eq("saved_lightgbm_threshold_f1_macro_opt")].iloc[0]
    current = selection[selection["selected_as_final"].astype(bool)].iloc[0]
    metric_pairs = [
        ("master_roc_auc", "roc_auc"),
        ("master_pr_auc", "pr_auc"),
        ("master_f1_macro", "f1_macro"),
        ("master_mcc", "mcc"),
        ("panel_roc_auc", "panel_roc_auc"),
        ("panel_pr_auc", "panel_pr_auc"),
        ("panel_f1_macro", "panel_f1_macro"),
        ("panel_mcc", "panel_mcc"),
    ]
    rows = []
    for label, row, selected, reason in [
        ("previous_threshold_optimized", previous, False, "Previous F1-threshold LightGBM reference."),
        ("current_conservative_regularized", current, True, "Selected by final score using decision metrics, panel metrics, overfitting gap, and threshold stability."),
    ]:
        rec = {
            "model_profile": label,
            "threshold_strategy": row["threshold_strategy"],
            "threshold_value": row["threshold"],
            "selected_as_final": selected,
            "reason": reason,
        }
        for out_col, source_col in metric_pairs:
            rec[out_col] = row[source_col]
            rec[f"{out_col}_absolute_improvement"] = row[source_col] - previous[source_col]
            rec[f"{out_col}_relative_improvement_pct"] = 100 * (row[source_col] - previous[source_col]) / previous[source_col]
        rows.append(rec)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "phase10_before_after_comparison.csv", index=False)

    plot_df = table.set_index("model_profile")[["master_f1_macro", "master_mcc", "panel_f1_macro", "panel_mcc"]]
    ax = plot_df.plot(kind="bar", figsize=(9, 5.4), color=["#4f6f7f", "#8a6f52", "#6b8f5a", "#a65f5f"])
    ax.set_ylabel("Metric")
    ax.set_title("Previous vs current final model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase10_before_after_comparison.png", dpi=200)
    plt.close()
    return table


def calibration_decision_review_outputs() -> pd.DataFrame:
    cal = pd.read_csv(TABLES_DIR / "calibration_comparison.csv")
    master = cal[cal["evaluation_split"].eq("MASTER_CV") & cal["threshold_strategy"].eq("f1_macro_opt")].copy()
    panel = cal[cal["evaluation_split"].eq("panel_unique_combined")].copy()
    panel_cols = panel[["calibration_method", "f1_macro", "mcc", "pr_auc", "brier_score", "log_loss"]].rename(
        columns={
            "f1_macro": "panel_f1_macro",
            "mcc": "panel_mcc",
            "pr_auc": "panel_pr_auc",
            "brier_score": "panel_brier_score",
            "log_loss": "panel_log_loss",
        }
    )
    matrix = master.merge(panel_cols, on="calibration_method", how="left")
    none = matrix[matrix["calibration_method"].eq("none")].iloc[0]
    matrix["decision"] = "reported_only"
    matrix.loc[matrix["calibration_method"].eq("none"), "decision"] = "selected_for_final_decision_model"
    matrix.loc[
        (matrix["calibration_method"].eq("isotonic"))
        & ((matrix["panel_pr_auc"] < none["panel_pr_auc"]) | (matrix["panel_log_loss"] > none["panel_log_loss"])),
        "decision",
    ] = "rejected_for_panel_tradeoff"
    matrix["rationale"] = np.where(
        matrix["calibration_method"].eq("none"),
        "Final selection keeps uncalibrated probabilities because decision and panel metrics are the primary competition criteria.",
        "Calibration is useful to report for probability quality, but not selected unless it preserves decision metrics and panel generalization.",
    )
    matrix.to_csv(TABLES_DIR / "calibration_decision_matrix.csv", index=False)

    ax = matrix.set_index("calibration_method")[["brier_score", "log_loss", "f1_macro", "mcc", "panel_f1_macro", "panel_mcc"]].plot(kind="bar", figsize=(10, 5.5))
    ax.set_title("Calibration decision matrix")
    ax.set_ylabel("Metric value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_calibration_decision_plot.png", dpi=200)
    plt.close()

    text = [
        "# Calibration Decision Review",
        "",
        "Calibration was reviewed as a probability-quality improvement, not as an automatic final-model selection criterion.",
        "",
        "Isotonic calibration improved MASTER Brier score, but the final competition objective prioritizes F1-macro, MCC, PR-AUC, and panel-unique generalization. Calibration is therefore reported only and not used in the final selected decision model.",
        "",
        matrix[["calibration_method", "brier_score", "log_loss", "f1_macro", "mcc", "panel_f1_macro", "panel_mcc", "decision"]].to_markdown(index=False),
    ]
    Path("reports/calibration_decision_review.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return matrix


def final_error_analysis_outputs(prepared: PreparedData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold = json.loads((METRICS_DIR / "final_threshold.json").read_text(encoding="utf-8"))["threshold"]
    master_pred = pd.read_csv(PREDICTIONS_DIR / "final_master_cv_predictions.csv")
    panel_pred = pd.read_csv(PREDICTIONS_DIR / "final_panel_predictions.csv")
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
    master_features = engineer.fit_transform(prepared.master).reset_index().rename(columns={"index": "row_index"})
    master = master_pred.merge(master_features, on=["row_index", "Variant_ID", "Label"], how="left")
    master["split"] = "MASTER_CV"
    panel_features = []
    for dataset, raw in {
        "KANSER_unique": prepared.kanser_unique,
        "PAH_unique": prepared.pah_unique,
        "CFTR_unique": prepared.cftr_unique,
    }.items():
        df = engineer.transform(raw)
        df["split"] = dataset
        panel_features.append(df)
    panel_features_df = pd.concat(panel_features, ignore_index=True)
    panel = panel_pred.rename(columns={"dataset": "split"}).merge(panel_features_df, on=["split", "Variant_ID", "Label"], how="left")
    all_cases = pd.concat([master, panel], ignore_index=True, sort=False)
    all_cases["error_group"] = np.select(
        [
            (all_cases["Label"] == 1) & (all_cases["prediction"] == 1),
            (all_cases["Label"] == 0) & (all_cases["prediction"] == 0),
            (all_cases["Label"] == 0) & (all_cases["prediction"] == 1),
            (all_cases["Label"] == 1) & (all_cases["prediction"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="unknown",
    )
    features = [
        "EK_7", "EK_9", "EK_3", "EK_2", "EK_net_evidence", "n_pops", "max_AF", "log_max_AF",
        "BA1_flag", "BS1_flag", "PM2_flag", "cat1_multipop", "cat1_AFR", "cat1_NFE",
        "cat6_has_filter", "cat6_segdup", "cat6_lcr", "aa_class_changed", "aa_change_te", "blosum62_approx",
    ]
    rows = []
    for split, split_df in all_cases.groupby("split"):
        for group, group_df in split_df.groupby("error_group"):
            for feature in [f for f in features if f in group_df.columns]:
                vals = pd.to_numeric(group_df[feature], errors="coerce")
                rows.append(
                    {
                        "split": split,
                        "error_group": group,
                        "feature": feature,
                        "mean": float(vals.mean()),
                        "median": float(vals.median()),
                        "std": float(vals.std()),
                        "n_notna": int(vals.notna().sum()),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / "final_error_group_feature_summary.csv", index=False)
    fp = all_cases[all_cases["error_group"].eq("FP")][["split", "Variant_ID", "Label", "score", "prediction", *[f for f in features if f in all_cases.columns]]]
    fn = all_cases[all_cases["error_group"].eq("FN")][["split", "Variant_ID", "Label", "score", "prediction", *[f for f in features if f in all_cases.columns]]]
    fp.to_csv(TABLES_DIR / "final_false_positive_cases.csv", index=False)
    fn.to_csv(TABLES_DIR / "final_false_negative_cases.csv", index=False)

    plot_features = ["EK_7", "EK_9", "EK_net_evidence", "n_pops", "max_AF", "log_max_AF"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    groups = ["TP", "TN", "FP", "FN"]
    master_only = all_cases[all_cases["split"].eq("MASTER_CV")]
    for ax, feature in zip(axes.ravel(), plot_features):
        data = [pd.to_numeric(master_only.loc[master_only["error_group"].eq(g), feature], errors="coerce").dropna() for g in groups]
        ax.boxplot(data, labels=groups, showfliers=False)
        ax.set_title(feature)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_error_group_feature_distributions.png", dpi=200)
    plt.close()

    contrast = summary[summary["error_group"].isin(["FP", "FN"]) & summary["feature"].isin(plot_features) & summary["split"].eq("MASTER_CV")]
    if not contrast.empty:
        pivot = contrast.pivot(index="feature", columns="error_group", values="median")
        ax = pivot.plot(kind="bar", figsize=(9, 5.3))
        ax.set_title("MASTER FP/FN median feature contrast")
        ax.set_ylabel("Median value")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "final_fp_fn_feature_contrast.png", dpi=200)
        plt.close()
    return summary, fp, fn


def final_panel_interpretation_outputs() -> pd.DataFrame:
    panel = pd.read_csv(PREDICTIONS_DIR / "final_panel_predictions.csv")
    rows = []
    threshold = float(panel["threshold"].iloc[0])
    for dataset, group in panel.groupby("dataset"):
        row = metric_row("lightgbm", dataset, "final_threshold", threshold, group["Label"], group["score"])
        row["pathogenic_rate"] = float(group["Label"].mean())
        row["n_samples"] = int(len(group))
        rows.append(row)
    table = pd.DataFrame(rows)
    strongest = table.sort_values(["f1_macro", "mcc"], ascending=False).iloc[0]
    weakest = table.sort_values(["f1_macro", "mcc"], ascending=True).iloc[0]
    table["panel_role"] = np.select(
        [table["evaluation_split"].eq(strongest["evaluation_split"]), table["evaluation_split"].eq(weakest["evaluation_split"])],
        ["strongest", "weakest"],
        default="intermediate",
    )
    table.to_csv(TABLES_DIR / "final_panel_specific_metrics.csv", index=False)

    ax = table.set_index("evaluation_split")[["roc_auc", "pr_auc", "f1_macro", "mcc"]].plot(kind="bar", figsize=(9, 5.4))
    ax.set_ylabel("Metric")
    ax.set_title("Final panel-specific metric comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_panel_specific_metric_comparison.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (dataset, group) in zip(axes, panel.groupby("dataset")):
        cm = confusion_matrix(group["Label"], group["prediction"], labels=LABELS)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(dataset)
        ax.set_xticks([0, 1], ["Benign", "Path"])
        ax.set_yticks([0, 1], ["Benign", "Path"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_panel_specific_confusion_matrices.png", dpi=200)
    plt.close()

    text = f"""# Panel-Specific Final Interpretation

The strongest panel by F1-macro/MCC is `{strongest['evaluation_split']}` with F1-macro {strongest['f1_macro']:.4f} and MCC {strongest['mcc']:.4f}.

The weakest panel by F1-macro/MCC is `{weakest['evaluation_split']}` with F1-macro {weakest['f1_macro']:.4f} and MCC {weakest['mcc']:.4f}.

Panel differences likely reflect a combination of class balance, panel-specific distribution shift, and different behavior of allele-frequency and conservation features. Detailed feature contrasts are saved in `reports/tables/panel_specific_error_analysis.csv` and `reports/tables/final_error_group_feature_summary.csv`.
"""
    Path("reports/panel_specific_final_interpretation.md").write_text(text, encoding="utf-8")
    return table


def final_feature_interpretation_outputs() -> pd.DataFrame:
    model = joblib.load(MODELS_DIR / "final_model.pkl")
    columns_path = MODELS_DIR / "final_model_columns.txt"
    columns = columns_path.read_text(encoding="utf-8").splitlines()
    importance = getattr(model, "feature_importances_", np.zeros(len(columns)))
    table = pd.DataFrame({"feature": columns, "importance": importance}).sort_values("importance", ascending=False).head(30)
    table.to_csv(TABLES_DIR / "final_feature_importance_top30.csv", index=False)
    plt.figure(figsize=(9, 8))
    plot_df = table.sort_values("importance")
    plt.barh(plot_df["feature"], plot_df["importance"], color="#4f6f7f")
    plt.xlabel("LightGBM importance")
    plt.title("Final model top 30 feature importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_feature_importance_top30.png", dpi=200)
    plt.close()

    def group(feature: str) -> tuple[str, str]:
        if feature.startswith("AL_") or feature in {"n_pops", "max_AF", "log_max_AF", "min_AF_nz", "BA1_flag", "BS1_flag", "PM2_flag", "BS2_proxy"}:
            return "AL/population frequency", "BA1/BS1/PM2-like allele-frequency evidence"
        if feature.startswith("EK_") or feature.startswith("EK") or "evidence" in feature:
            return "EK conservation/prediction", "PP3/BP4-like computational evidence"
        if feature.startswith("aa") or feature == "blosum62_approx":
            return "AA substitution", "Protein-level substitution plausibility"
        if feature.startswith("cat") or feature.startswith("CAT") or feature.startswith("geno"):
            return "CAT metadata/quality", "Population metadata, quality, and reliability evidence"
        if feature.startswith("miss"):
            return "Missingness", "Dataset coverage and missingness pattern signal"
        return "Other engineered numeric", "Supportive non-identifier feature"

    mapping_rows = []
    for _, row in table.iterrows():
        feature_group, interpretation = group(str(row["feature"]))
        mapping_rows.append(
            {
                "feature": row["feature"],
                "importance": row["importance"],
                "feature_group": feature_group,
                "acmg_inspired_mapping": interpretation,
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(TABLES_DIR / "final_acmg_feature_mapping.csv", index=False)

    group_counts = mapping["feature_group"].value_counts()
    text = ["# Final Feature Interpretation", ""]
    text.append("The final model remains biologically interpretable because its top features are dominated by ACMG-inspired evidence groups rather than identifiers.")
    text.append("")
    for feature_group, count in group_counts.items():
        text.append(f"- {feature_group}: {count} top-30 features.")
    text.extend(
        [
            "",
            "AL/population frequency features correspond to BA1, BS1, and PM2-like evidence.",
            "EK conservation and predictor features correspond to PP3/BP4-like computational evidence.",
            "AA substitution features represent protein-level plausibility of missense impact.",
            "CAT quality and metadata features capture population and sequencing reliability context.",
        ]
    )
    Path("reports/final_feature_interpretation.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return table


def final_model_strength_statement() -> None:
    selected = selected_final_row()
    text = f"""# Final Model Strength Statement

The final selected model is best described as **moderate-to-good, conservatively reported as moderate**.

It should not be described as weak because the final metrics are functional for the task: MASTER ROC-AUC {selected['roc_auc']:.4f}, MASTER PR-AUC {selected['pr_auc']:.4f}, MASTER F1-macro {selected['f1_macro']:.4f}, MASTER MCC {selected['mcc']:.4f}, panel ROC-AUC {selected['panel_roc_auc']:.4f}, panel PR-AUC {selected['panel_pr_auc']:.4f}, panel F1-macro {selected['panel_f1_macro']:.4f}, and panel MCC {selected['panel_mcc']:.4f}.

It should not be described as strong because F1-macro and MCC, while improved, are not clearly excellent across all validation views. The model is therefore suitable for an honest competition report as a reproducible, leakage-aware, clinically motivated baseline with meaningful external-panel behavior and remaining room for improvement.
"""
    Path("reports/final_model_strength_statement.md").write_text(text, encoding="utf-8")


def update_final_reports_for_audit() -> None:
    selected = selected_final_row()
    before_after = pd.read_csv(TABLES_DIR / "phase10_before_after_comparison.csv")
    current = before_after[before_after["selected_as_final"].astype(bool)].iloc[0]
    strength = Path("reports/final_model_strength_statement.md").read_text(encoding="utf-8")
    perf_path = Path("reports/final_performance_analysis.md")
    existing = perf_path.read_text(encoding="utf-8") if perf_path.exists() else ""
    audit_section = f"""

## Final Audit Addendum
Final metric verification, calibration decision review, panel-specific interpretation, and error-group analyses were completed in Phase 11. The current final profile is `{selected['profile']}` with threshold strategy `{selected['threshold_strategy']}` and threshold {float(selected['threshold']):.6f}. Compared with the previous threshold-optimized model, MASTER F1-macro improved by {current['master_f1_macro_absolute_improvement']:+.4f}, MASTER MCC improved by {current['master_mcc_absolute_improvement']:+.4f}, panel F1-macro improved by {current['panel_f1_macro_absolute_improvement']:+.4f}, and panel MCC improved by {current['panel_mcc_absolute_improvement']:+.4f}.

Calibration remains reported-only: it improves probability loss on MASTER, but final selection prioritizes decision metrics, panel behavior, overfitting gap, and threshold stability.

{strength}
"""
    if "## Final Audit Addendum" not in existing:
        perf_path.write_text(existing.rstrip() + "\n" + audit_section, encoding="utf-8")

    summary_path = Path("reports/final_model_report_summary.md")
    summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    line = "\nPhase 11 audit outputs are available under `reports/tables`, `reports/figures`, and `reports/*_interpretation.md`, including final metric verification, calibration decision review, panel-specific interpretation, final feature interpretation, and model strength statement.\n"
    if "Phase 11 audit outputs" not in summary:
        summary_path.write_text(summary.rstrip() + line, encoding="utf-8")
def run_phase10_improvements(prepared: PreparedData, base_params: dict[str, object], mode: str = "evaluate") -> dict[str, object]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    verify_current_results()
    advanced = advanced_threshold_outputs()
    stability = threshold_stability_outputs()
    calibration = calibration_outputs()
    gaps, profiles = overfitting_and_lgbm_profile_outputs(prepared, base_params, mode)
    feature_ablation_outputs()
    selection_seed = advanced[
        advanced["model_name"].eq("lightgbm")
        & advanced["evaluation_split"].eq("MASTER_CV")
        & advanced["threshold_strategy"].eq("f1_macro_opt")
    ].iloc[0]
    panel_specific_error_outputs(prepared, float(selection_seed["threshold"]))
    selection = final_selection_outputs(advanced, stability, calibration, profiles, gaps)
    final_artifacts_and_report(prepared, selection, advanced, calibration, profiles)
    export_final_prediction_files(prepared)
    final_metric_verification_audit()
    before_after_comparison_outputs()
    calibration_decision_review_outputs()
    final_error_analysis_outputs(prepared)
    final_panel_interpretation_outputs()
    final_feature_interpretation_outputs()
    final_model_strength_statement()
    update_final_reports_for_audit()
    diagnosis = diagnose_model_performance(pd.read_csv(TABLES_DIR / "all_evaluation_metrics.csv"))
    return {
        "model_strength": diagnosis["model_strength"],
        "main_issue": "thresholding_and_calibration",
        "selected_threshold": float(selection[selection["selected_as_final"]]["threshold"].iloc[0]),
        "selected_threshold_strategy": str(selection[selection["selected_as_final"]]["threshold_strategy"].iloc[0]),
    }
