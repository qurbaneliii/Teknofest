from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def classify_error_cases(
    predictions: pd.DataFrame,
    probability_column: str,
    threshold: float,
    source: str,
    uncertainty_margin: float = 0.05,
) -> pd.DataFrame:
    required = {"Variant_ID", "Label", probability_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["source"] = source
    frame["probability"] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame["threshold"] = float(threshold)
    frame["predicted_label"] = (frame["probability"] >= threshold).astype(int)
    frame["error_type"] = np.select(
        [
            frame["Label"].eq(1) & frame["predicted_label"].eq(0),
            frame["Label"].eq(0) & frame["predicted_label"].eq(1),
            frame["Label"].eq(1) & frame["predicted_label"].eq(1),
        ],
        ["false_negative", "false_positive", "true_positive"],
        default="true_negative",
    )
    frame["uncertainty_flag"] = np.select(
        [
            frame["probability"].sub(threshold).abs().le(uncertainty_margin),
            frame["probability"].ge(threshold + uncertainty_margin),
        ],
        ["uncertain", "confident_pathogenic"],
        default="confident_benign",
    )
    frame["error_hypothesis"] = np.select(
        [
            frame["error_type"].isin(["false_negative", "false_positive"]) & frame["uncertainty_flag"].eq("uncertain"),
            frame["source"].ne("MASTER_CV") & frame["error_type"].isin(["false_negative", "false_positive"]),
            frame["error_type"].eq("false_negative") & frame["probability"].lt(threshold - uncertainty_margin),
            frame["error_type"].eq("false_positive") & frame["probability"].gt(threshold + uncertainty_margin),
        ],
        ["threshold_boundary", "panel_shift", "feature_or_capacity", "calibration_or_feature"],
        default="correct_or_not_applicable",
    )
    return frame


def merge_error_features(errors: pd.DataFrame, raw_features: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw_features is None or "Variant_ID" not in raw_features.columns:
        return errors
    keep = [column for column in raw_features.columns if column == "Variant_ID" or column in {"AA_1", "AA_2", "CAT_1", "CAT_2", "EK_2", "EK_3", "EK_4", "EK_6", "EK_7", "EK_9", "AL_1", "AL_2"}]
    return errors.merge(raw_features[keep].drop_duplicates("Variant_ID"), on="Variant_ID", how="left")


def error_pattern_summary(errors: pd.DataFrame) -> pd.DataFrame:
    wrong = errors[errors["error_type"].isin(["false_negative", "false_positive"])].copy()
    if wrong.empty:
        return pd.DataFrame(columns=["pattern_type", "pattern", "error_type", "n_cases", "mean_probability", "likely_issue"])
    rows: list[pd.DataFrame] = []
    if {"AA_1", "AA_2"}.issubset(wrong.columns):
        aa = wrong.assign(pattern=wrong["AA_1"].astype(str) + ">" + wrong["AA_2"].astype(str)).groupby(["pattern", "error_type"], as_index=False).agg(n_cases=("Variant_ID", "size"), mean_probability=("probability", "mean"))
        aa["pattern_type"] = "amino_acid_substitution"
        rows.append(aa)
    if "panel" in wrong.columns:
        panel = wrong.groupby(["panel", "error_type"], as_index=False).agg(n_cases=("Variant_ID", "size"), mean_probability=("probability", "mean")).rename(columns={"panel": "pattern"})
        panel["pattern_type"] = "panel"
        rows.append(panel)
    if "error_hypothesis" in wrong.columns:
        cause = wrong.groupby(["error_hypothesis", "error_type"], as_index=False).agg(n_cases=("Variant_ID", "size"), mean_probability=("probability", "mean")).rename(columns={"error_hypothesis": "pattern"})
        cause["pattern_type"] = "hypothesis"
        rows.append(cause)
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if summary.empty:
        return summary
    summary["likely_issue"] = np.select(
        [summary["pattern_type"].eq("panel"), summary["pattern"].eq("threshold_boundary")],
        ["panel_shift", "threshold_issue"],
        default="feature_capacity_or_ambiguous_region",
    )
    return summary.sort_values(["n_cases", "pattern_type"], ascending=[False, True])


def save_error_analysis(
    master_errors: pd.DataFrame,
    panel_errors: pd.DataFrame,
    reports_dir: str | Path = "reports",
) -> dict[str, Path]:
    reports = Path(reports_dir)
    tables = reports / "tables"
    figures = reports / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    errors = pd.concat([master_errors, panel_errors], ignore_index=True)
    false_negative = errors[errors["error_type"].eq("false_negative")]
    false_positive = errors[errors["error_type"].eq("false_positive")]
    high_confidence = errors[
        errors["error_type"].isin(["false_negative", "false_positive"])
        & errors["uncertainty_flag"].ne("uncertain")
    ]
    summary = error_pattern_summary(errors)
    false_negative.to_csv(tables / "final_false_negative_analysis.csv", index=False)
    false_positive.to_csv(tables / "final_false_positive_analysis.csv", index=False)
    high_confidence.to_csv(tables / "high_confidence_error_cases.csv", index=False)
    summary.to_csv(tables / "error_pattern_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    for name, frame in errors.groupby("error_type"):
        plt.hist(frame["probability"], bins=20, alpha=0.45, label=name)
    plt.xlabel("Predicted pathogenic probability")
    plt.ylabel("Case count")
    plt.legend()
    plt.tight_layout()
    plot_path = figures / "error_probability_distributions.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()
    report = [
        "# Final Error Analysis",
        "",
        f"False negatives: {len(false_negative)}. False positives: {len(false_positive)}. High-confidence errors: {len(high_confidence)}.",
        "",
        "Boundary errors are candidates for threshold review. Errors far from the threshold are retained as feature, capacity, calibration, panel-shift, or clinically ambiguous cases; no label-derived correction is applied.",
    ]
    (reports / "final_error_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "false_negative": tables / "final_false_negative_analysis.csv",
        "false_positive": tables / "final_false_positive_analysis.csv",
        "high_confidence": tables / "high_confidence_error_cases.csv",
        "summary": tables / "error_pattern_summary.csv",
    }

