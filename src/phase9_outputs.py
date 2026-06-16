from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import FIGURES_DIR, PREDICTIONS_DIR, TABLES_DIR
from teknofest.data_prep import PreparedData
from teknofest.features import FeatureEngineer, detect_binary_al_cols
from teknofest.training import acmg_rule_probability, fit_lr_ek, fold_engineered_data
from teknofest.validation import best_f1_macro_threshold, contamination_aware_folds, youden_j_threshold


FIG_DPI = 300
LABELS = [0, 1]
DISPLAY_LABELS = ["Benign", "Pathogenic"]


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def _safe_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7), labels=LABELS))


def _metric_row(
    model_name: str,
    evaluation_split: str,
    threshold_type: str,
    threshold_value: float,
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = (score >= threshold_value).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=LABELS).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "model_name": model_name,
        "evaluation_split": evaluation_split,
        "threshold_type": threshold_type,
        "threshold_value": float(threshold_value),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "roc_auc": _safe_auc(y, score),
        "pr_auc": _safe_ap(y, score),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity) if pd.notna(specificity) else np.nan,
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 and len(np.unique(y)) > 1 else 0.0,
        "log_loss": _safe_log_loss(y, score),
        "brier_score": float(brier_score_loss(y, np.clip(score, 0, 1))),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(y)),
    }


def _thresholds_for(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> list[tuple[str, float]]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if len(np.unique(y)) < 2:
        return [("default_0.5", 0.5)]
    f1_thr, _ = best_f1_macro_threshold(y, score)
    youden_thr, _ = youden_j_threshold(y, score)
    return [("default_0.5", 0.5), ("f1_macro_opt", float(f1_thr)), ("youden_j", float(youden_thr))]


def _collect_baseline_oof(prepared: PreparedData) -> pd.DataFrame:
    rows = []
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    ek_cols = [f"EK_{i}" for i in range(1, 10)]
    for fold in folds:
        train_df, val_df = fold_engineered_data(prepared, fold.train_idx, fold.val_idx)
        y_train = train_df["Label"]
        majority_score = pd.Series(float(y_train.mean() >= 0.5), index=val_df.index)

        lr = fit_lr_ek()
        lr.fit(train_df[ek_cols], y_train)
        lr_score = lr.predict_proba(val_df[ek_cols])[:, 1]

        for model_name, score in [
            ("majority_baseline", majority_score),
            ("acmg_rule_baseline", acmg_rule_probability(val_df)),
            ("logistic_regression_ek_only", pd.Series(lr_score, index=val_df.index)),
        ]:
            rows.append(
                pd.DataFrame(
                    {
                        "model_name": model_name,
                        "fold": fold.fold,
                        "row_index": val_df.index,
                        "Variant_ID": val_df["Variant_ID"].to_numpy(),
                        "Label": val_df["Label"].to_numpy(),
                        "score": np.asarray(score, dtype=float),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _collect_master_model_predictions(prepared: PreparedData) -> pd.DataFrame:
    frames = [_collect_baseline_oof(prepared)]
    oof_path = PREDICTIONS_DIR / "oof_predictions.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        frames.append(
            pd.DataFrame(
                {
                    "model_name": "lightgbm",
                    "fold": oof["fold"],
                    "row_index": oof["row_index"],
                    "Variant_ID": oof["Variant_ID"],
                    "Label": oof["Label"],
                    "score": oof["lightgbm_probability"],
                }
            )
        )
    stack_path = Path("reports/master_prompt/l0_stack_oof_predictions.csv")
    if stack_path.exists():
        stack = pd.read_csv(stack_path)
        model_columns = {
            "catboost_probability": "catboost",
            "xgboost_probability": "xgboost",
            "extra_trees_probability": "extra_trees",
            "lr_ek_only_probability": "logistic_regression_ek_only_stack_run",
            "stack_probability": "ensemble_stack_l1",
        }
        for col, model_name in model_columns.items():
            if col in stack.columns:
                frames.append(
                    pd.DataFrame(
                        {
                            "model_name": model_name,
                            "fold": stack["fold"],
                            "row_index": stack["row_index"],
                            "Variant_ID": stack["Variant_ID"],
                            "Label": stack["Label"],
                            "score": stack[col],
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def _panel_predictions() -> pd.DataFrame:
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


def save_all_evaluation_metrics(prepared: PreparedData) -> pd.DataFrame:
    rows = []
    master_preds = _collect_master_model_predictions(prepared)
    master_preds.to_csv(PREDICTIONS_DIR / "master_cv_model_predictions_phase9.csv", index=False)
    for model_name, group in master_preds.groupby("model_name"):
        for threshold_type, threshold in _thresholds_for(group["Label"], group["score"]):
            rows.append(_metric_row(model_name, "MASTER_CV", threshold_type, threshold, group["Label"], group["score"]))

    panel = _panel_predictions()
    panel.to_csv(PREDICTIONS_DIR / "panel_unique_model_predictions_phase9.csv", index=False)
    for split, group in panel.groupby("evaluation_split"):
        threshold = float(group["saved_threshold"].iloc[0])
        rows.append(_metric_row("lightgbm", split, "saved_panel_threshold", threshold, group["Label"], group["score"]))
    threshold = float(panel["saved_threshold"].median())
    rows.append(
        _metric_row(
            "lightgbm",
            "panel_unique_combined",
            "saved_panel_threshold",
            threshold,
            panel["Label"],
            panel["score"],
        )
    )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES_DIR / "all_evaluation_metrics.csv", index=False)
    return metrics


def _draw_confusion(y_true, y_score, threshold: float, title: str, out_path: Path) -> None:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=LABELS)
    denom = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
    plt.figure(figsize=(6.2, 5.2))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xticks([0, 1], DISPLAY_LABELS)
    plt.yticks([0, 1], DISPLAY_LABELS)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]}\n{pct[i, j] * 100:.1f}%", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


def save_confusion_matrices() -> None:
    oof = pd.read_csv(PREDICTIONS_DIR / "oof_predictions.csv")
    f1_thr, _ = best_f1_macro_threshold(oof["Label"], oof["lightgbm_probability"])
    _draw_confusion(
        oof["Label"],
        oof["lightgbm_probability"],
        f1_thr,
        f"MASTER CV confusion matrix (threshold={f1_thr:.3f})",
        FIGURES_DIR / "confusion_matrix_master.png",
    )
    panel = pd.read_csv(PREDICTIONS_DIR / "panel_unique_predictions.csv")
    for dataset, filename in [
        ("KANSER_unique", "confusion_matrix_kanser_unique.png"),
        ("PAH_unique", "confusion_matrix_pah_unique.png"),
        ("CFTR_unique", "confusion_matrix_cftr_unique.png"),
    ]:
        group = panel[panel["dataset"].eq(dataset)]
        if not group.empty:
            _draw_confusion(
                group["Label"],
                group["predicted_probability"],
                float(group["threshold"].iloc[0]),
                f"{dataset} confusion matrix",
                FIGURES_DIR / filename,
            )
    _draw_confusion(
        panel["Label"],
        panel["predicted_probability"],
        float(panel["threshold"].median()),
        "Panel-unique combined confusion matrix",
        FIGURES_DIR / "confusion_matrix_panel_unique_combined.png",
    )


def save_roc_pr_curves() -> None:
    oof = pd.read_csv(PREDICTIONS_DIR / "oof_predictions.csv")
    y = oof["Label"].to_numpy()
    score = oof["lightgbm_probability"].to_numpy()
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, label=f"MASTER CV AUC={auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("MASTER CV ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_master.png", dpi=FIG_DPI)
    plt.close()

    precision, recall, _ = precision_recall_curve(y, score)
    ap = average_precision_score(y, score)
    plt.figure(figsize=(7, 5.5))
    plt.plot(recall, precision, label=f"MASTER CV AP={ap:.3f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("MASTER CV precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve_master.png", dpi=FIG_DPI)
    plt.close()

    panel = pd.read_csv(PREDICTIONS_DIR / "panel_unique_predictions.csv")
    plt.figure(figsize=(7, 5.5))
    for dataset, group in panel.groupby("dataset"):
        if group["Label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(group["Label"], group["predicted_probability"])
        auc = roc_auc_score(group["Label"], group["predicted_probability"])
        plt.plot(fpr, tpr, label=f"{dataset} AUC={auc:.3f}", linewidth=2)
    fpr, tpr, _ = roc_curve(panel["Label"], panel["predicted_probability"])
    auc = roc_auc_score(panel["Label"], panel["predicted_probability"])
    plt.plot(fpr, tpr, label=f"Combined AUC={auc:.3f}", linewidth=2.5, color="black")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Panel-unique ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_panel_unique.png", dpi=FIG_DPI)
    plt.close()

    plt.figure(figsize=(7, 5.5))
    for dataset, group in panel.groupby("dataset"):
        if group["Label"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(group["Label"], group["predicted_probability"])
        ap = average_precision_score(group["Label"], group["predicted_probability"])
        plt.plot(recall, precision, label=f"{dataset} AP={ap:.3f}", linewidth=2)
    precision, recall, _ = precision_recall_curve(panel["Label"], panel["predicted_probability"])
    ap = average_precision_score(panel["Label"], panel["predicted_probability"])
    plt.plot(recall, precision, label=f"Combined AP={ap:.3f}", linewidth=2.5, color="black")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Panel-unique precision-recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve_panel_unique.png", dpi=FIG_DPI)
    plt.close()


def save_threshold_optimization() -> pd.DataFrame:
    oof = pd.read_csv(PREDICTIONS_DIR / "oof_predictions.csv")
    y = oof["Label"].to_numpy()
    score = oof["lightgbm_probability"].to_numpy()
    f1_thr, _ = best_f1_macro_threshold(y, score)
    youden_thr, _ = youden_j_threshold(y, score)
    rows = []
    for threshold in np.linspace(0.0, 1.0, 201):
        pred = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=LABELS).ravel()
        recall = recall_score(y, pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append(
            {
                "threshold": threshold,
                "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall,
                "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
                "youden_j": recall + specificity - 1 if pd.notna(specificity) else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "threshold_optimization.csv", index=False)
    plt.figure(figsize=(9, 5.8))
    for col in ["f1_macro", "precision", "recall", "mcc", "youden_j"]:
        plt.plot(table["threshold"], table[col], label=col)
    for name, threshold, color in [
        ("default 0.5", 0.5, "gray"),
        ("F1-opt", f1_thr, "#9b4d3f"),
        ("Youden-J", youden_thr, "#3f6f4f"),
    ]:
        plt.axvline(threshold, linestyle="--", linewidth=1.5, color=color, label=f"{name}={threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("MASTER CV threshold optimization")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_optimization.png", dpi=FIG_DPI)
    plt.close()
    return table


def save_model_comparison(metrics: pd.DataFrame) -> None:
    subset = metrics[metrics["evaluation_split"].eq("MASTER_CV") & metrics["threshold_type"].eq("f1_macro_opt")].copy()
    if subset.empty:
        subset = metrics[metrics["evaluation_split"].eq("MASTER_CV") & metrics["threshold_type"].eq("default_0.5")].copy()
    cols = ["roc_auc", "pr_auc", "f1_macro", "mcc"]
    plot_df = subset.set_index("model_name")[cols].sort_values("roc_auc", ascending=False)
    plot_df.to_csv(TABLES_DIR / "model_comparison_metrics.csv")
    ax = plot_df.plot(kind="bar", figsize=(11, 6), width=0.82)
    ax.set_ylabel("Metric value")
    ax.set_xlabel("Model")
    ax.set_title("MASTER CV model comparison")
    ax.legend(title="Metric", ncol=2)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_metrics.png", dpi=FIG_DPI)
    plt.close()


def save_feature_importance_top30() -> None:
    src = TABLES_DIR / "feature_importance.csv"
    if not src.exists():
        return
    importance = pd.read_csv(src).sort_values("mean_abs_shap", ascending=False).head(30)
    importance.to_csv(TABLES_DIR / "feature_importance_top30.csv", index=False)
    plt.figure(figsize=(9, 9))
    plot_df = importance.sort_values("mean_abs_shap")
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#4f6f7f")
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Top 30 feature importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_top30.png", dpi=FIG_DPI)
    plt.close()


def save_correlation_matrix(prepared: PreparedData) -> None:
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
    master = engineer.fit_transform(prepared.master)
    selected = []
    importance_path = TABLES_DIR / "feature_importance.csv"
    if importance_path.exists():
        selected.extend(pd.read_csv(importance_path)["feature"].head(20).tolist())
    selected.extend(
        [
            "EK_1",
            "EK_2",
            "EK_3",
            "EK_4",
            "EK_5",
            "EK_6",
            "EK_7",
            "EK_8",
            "EK_9",
            "EK7xEK9",
            "n_pops",
            "max_AF",
            "log_max_AF",
            "min_AF_nz",
            "n_nonmiss_AL",
            "BA1_flag",
            "BS1_flag",
            "PM2_flag",
            "BS2_proxy",
            "EK_net_evidence",
            "aa_class_changed",
            "aa_change_te",
        ]
    )
    cols = []
    for col in selected:
        if col in master.columns and pd.api.types.is_numeric_dtype(master[col]) and col not in cols:
            cols.append(col)
    cols = cols[:35]
    corr = master[cols].corr(method="spearman")
    corr.to_csv(TABLES_DIR / "correlation_matrix_top_features.csv")
    plt.figure(figsize=(max(9, len(cols) * 0.35), max(8, len(cols) * 0.32)))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman correlation")
    plt.xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(len(cols)), cols, fontsize=8)
    plt.title("Correlation matrix for top numeric features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix_top_features.png", dpi=FIG_DPI)
    plt.close()


def save_class_distribution(prepared: PreparedData) -> None:
    rows = []
    datasets = {
        "MASTER": prepared.master,
        "KANSER": prepared.kanser,
        "PAH": prepared.pah,
        "CFTR": prepared.cftr,
        "KANSER_unique": prepared.kanser_unique,
        "PAH_unique": prepared.pah_unique,
        "CFTR_unique": prepared.cftr_unique,
    }
    for name, df in datasets.items():
        counts = df["Label"].value_counts().reindex(LABELS, fill_value=0)
        for label in LABELS:
            rows.append({"dataset": name, "label": DISPLAY_LABELS[label], "count": int(counts.loc[label])})
    dist = pd.DataFrame(rows)
    dist.to_csv(TABLES_DIR / "class_distribution_by_dataset.csv", index=False)
    pivot = dist.pivot(index="dataset", columns="label", values="count").loc[list(datasets.keys())]
    ax = pivot.plot(kind="bar", stacked=False, figsize=(10, 5.8), color=["#4f6f7f", "#b56557"])
    ax.set_ylabel("Variant count")
    ax.set_xlabel("Dataset")
    ax.set_title("Class distribution by dataset")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution_by_dataset.png", dpi=FIG_DPI)
    plt.close()


def save_missingness_by_group(prepared: PreparedData) -> None:
    datasets = {
        "MASTER": prepared.master,
        "KANSER": prepared.kanser,
        "PAH": prepared.pah,
        "CFTR": prepared.cftr,
    }
    groups = {
        "AL": lambda df: [c for c in df.columns if c.startswith("AL_")],
        "EK": lambda df: [c for c in df.columns if c.startswith("EK_")],
        "CAT": lambda df: [c for c in df.columns if c.startswith("CAT_")],
        "AA": lambda df: [c for c in df.columns if c.startswith("AA_")],
    }
    rows = []
    for dataset, df in datasets.items():
        for group_name, selector in groups.items():
            columns = selector(df)
            rows.append(
                {
                    "dataset": dataset,
                    "feature_group": group_name,
                    "missing_rate": float(df[columns].isna().mean().mean()) if columns else np.nan,
                    "columns": len(columns),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "missingness_by_feature_group.csv", index=False)
    pivot = table.pivot(index="dataset", columns="feature_group", values="missing_rate")
    ax = pivot.plot(kind="bar", figsize=(9, 5.5), color=["#4f6f7f", "#b56557", "#6c8f5f", "#8b7b52"])
    ax.set_ylabel("Mean missingness rate")
    ax.set_xlabel("Dataset")
    ax.set_title("Missingness by feature group")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missingness_by_feature_group.png", dpi=FIG_DPI)
    plt.close()


def save_error_analysis_plot(prepared: PreparedData) -> None:
    oof = pd.read_csv(PREDICTIONS_DIR / "oof_predictions.csv")
    f1_thr, _ = best_f1_macro_threshold(oof["Label"], oof["lightgbm_probability"])
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
    master = engineer.fit_transform(prepared.master)
    preds = oof[["Variant_ID", "Label", "lightgbm_probability"]].copy()
    preds["prediction"] = (preds["lightgbm_probability"] >= f1_thr).astype(int)
    preds["error_group"] = np.select(
        [
            (preds["Label"] == 1) & (preds["prediction"] == 1),
            (preds["Label"] == 0) & (preds["prediction"] == 0),
            (preds["Label"] == 0) & (preds["prediction"] == 1),
            (preds["Label"] == 1) & (preds["prediction"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="unknown",
    )
    merged = preds.merge(master, on=["Variant_ID", "Label"], how="left")
    features = [f for f in ["EK_7", "EK_9", "n_pops", "max_AF", "log_max_AF", "EK_net_evidence"] if f in merged.columns]
    rows = []
    for feature in features:
        for group, data in merged.groupby("error_group"):
            rows.append(
                {
                    "feature": feature,
                    "error_group": group,
                    "mean": float(pd.to_numeric(data[feature], errors="coerce").mean()),
                    "median": float(pd.to_numeric(data[feature], errors="coerce").median()),
                    "n_notna": int(pd.to_numeric(data[feature], errors="coerce").notna().sum()),
                }
            )
    pd.DataFrame(rows).to_csv(TABLES_DIR / "error_analysis_key_features.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    order = ["TP", "TN", "FP", "FN"]
    for ax, feature in zip(axes.ravel(), features):
        data = [pd.to_numeric(merged.loc[merged["error_group"].eq(group), feature], errors="coerce").dropna() for group in order]
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_title(feature)
        ax.set_xlabel("Prediction group")
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    fig.suptitle("Error analysis by key features", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_analysis_key_features.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def generate_phase9_outputs(prepared: PreparedData) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = save_all_evaluation_metrics(prepared)
    save_correlation_matrix(prepared)
    save_confusion_matrices()
    save_roc_pr_curves()
    save_threshold_optimization()
    save_model_comparison(metrics)
    save_feature_importance_top30()
    save_class_distribution(prepared)
    save_missingness_by_group(prepared)
    save_error_analysis_plot(prepared)
