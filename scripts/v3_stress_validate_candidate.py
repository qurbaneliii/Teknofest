"""Repeated contamination-safe stress validation for one exploratory V3 candidate."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from teknofest_v3.data import load_labeled_competition_data
from teknofest_v3.features import V3FeatureBuilder
from teknofest_v3.metrics import binary_metrics, choose_threshold
from teknofest_v3.selection import robust_genomics_score

BASELINE = {"master_f1_macro": 0.7764, "master_mcc": 0.5548, "panel_f1_macro": 0.7708, "panel_mcc": 0.5825, "threshold": 0.471}


def hgb() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=3.0, random_state=42))])


def append(rows: list[dict], y, p, threshold: float, **meta) -> dict:
    row = binary_metrics(y, p, threshold)
    row.update(meta)
    rows.append(row)
    return row


def bootstrap_mean(values: pd.Series, seed: int = 42, n_bootstrap: int = 2000) -> tuple[float, float]:
    values = values.dropna().to_numpy(float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["f1_macro", "mcc", "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "overfitting_gap_f1_macro"]
    rows = []
    for keys, group in frame.groupby(["evaluation_split", "threshold_source"], dropna=False):
        row = {"evaluation_split": keys[0], "threshold_source": keys[1], "n_folds": len(group)}
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            low, high = bootstrap_mean(values)
            row.update({f"{metric}_mean": values.mean(), f"{metric}_std": values.std(ddof=1), f"{metric}_min": values.min(), f"{metric}_max": values.max(), f"{metric}_median": values.median(), f"{metric}_q1": values.quantile(0.25), f"{metric}_q3": values.quantile(0.75), f"{metric}_bootstrap_ci_low": low, f"{metric}_bootstrap_ci_high": high})
        rows.append(row)
    return pd.DataFrame(rows)


def make_report(reports: Path, aggregate_metrics: pd.DataFrame, threshold: pd.DataFrame, decision: dict) -> None:
    master = aggregate_metrics[(aggregate_metrics.evaluation_split.eq("MASTER_validation")) & (aggregate_metrics.threshold_source.eq("validation_f1_macro"))].iloc[0]
    panels = aggregate_metrics[(aggregate_metrics.evaluation_split.eq("panel_combined")) & (aggregate_metrics.threshold_source.eq("validation_f1_macro"))].iloc[0]
    kans = aggregate_metrics[(aggregate_metrics.evaluation_split.eq("KANSER")) & (aggregate_metrics.threshold_source.eq("validation_f1_macro"))].iloc[0]
    pah = aggregate_metrics[(aggregate_metrics.evaluation_split.eq("PAH")) & (aggregate_metrics.threshold_source.eq("validation_f1_macro"))].iloc[0]
    val_threshold = threshold[threshold.threshold_source.eq("validation_f1_macro")].iloc[0]
    text = f"""# HistGradientBoosting Candidate Stress Validation

## Executive summary

Candidate: `hist_gradient_boosting` with `v3_safe_minimal`. This is repeated internal validation, **not official hidden-test performance**. Every builder and model was fitted only on each fold's MASTER training rows; panels were evaluation-only.

## Protocol

Five stratified folds were repeated over five seeds (25 fold validations). Thresholds 0.50, 0.471, validation F1-macro, and validation MCC were evaluated. Validation-derived thresholds were selected from validation scores only and never from panels.

## Repeated MASTER validation

At validation-F1 thresholds: F1-macro {master['f1_macro_mean']:.4f} ± {master['f1_macro_std']:.4f}; MCC {master['mcc_mean']:.4f} ± {master['mcc_std']:.4f}; ROC-AUC {master['roc_auc_mean']:.4f}; PR-AUC {master['pr_auc_mean']:.4f}.

## Panel stress result

Combined panels: F1-macro {panels['f1_macro_mean']:.4f}; MCC {panels['mcc_mean']:.4f}. KANSER: F1-macro {kans['f1_macro_mean']:.4f}; MCC {kans['mcc_mean']:.4f}. PAH: F1-macro {pah['f1_macro_mean']:.4f}; MCC {pah['mcc_mean']:.4f}.

## Threshold stability

Validation F1 threshold mean {val_threshold['mean']:.4f}, std {val_threshold['std']:.4f}, IQR {val_threshold['iqr']:.4f}; range {val_threshold['min']:.4f}–{val_threshold['max']:.4f}.

## Baseline comparison and decision

Protected baseline reference is MASTER OOF F1-macro 0.7764/MCC 0.5548 and panel-unique F1-macro 0.7708/MCC 0.5825. The protocol is not identical, so this comparison is directional only. Decision: `{decision['final_decision']}`. {decision['reason']}

The protected LightGBM baseline remains final. HistGradientBoosting remains exploratory. No official hidden-test metric is claimed.

## Reproduce

`python scripts/v3_stress_validate_candidate.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main --output-dir reports/v3/stress_validation --model hist_gradient_boosting --feature-set v3_safe_minimal --n-splits 5 --seeds 42,2026,777,123,999 --threshold-strategy validation_f1 --compare-baseline yes`
"""
    (reports / "stress_validation_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--output-dir", default="reports/v3/stress_validation")
    parser.add_argument("--model", default="hist_gradient_boosting"); parser.add_argument("--feature-set", default="v3_safe_minimal")
    parser.add_argument("--n-splits", type=int, default=5); parser.add_argument("--seeds", default="42,2026,777,123,999")
    parser.add_argument("--threshold-strategy", default="validation_f1"); parser.add_argument("--compare-baseline", default="yes")
    args = parser.parse_args()
    if args.model != "hist_gradient_boosting" or args.feature_set != "v3_safe_minimal":
        raise ValueError("This controlled script supports only hist_gradient_boosting + v3_safe_minimal.")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    reports, tables = ROOT / args.output_dir, ROOT / args.output_dir / "tables"
    artifacts = ROOT / "artifacts" / "v3" / "stress_validation"
    for path in (tables, artifacts / "predictions", artifacts / "models", artifacts / "builders"):
        path.mkdir(parents=True, exist_ok=True)
    data = load_labeled_competition_data(args.data_dir); master = data["MASTER"].reset_index(drop=True)
    panel_variant_ids = set(pd.concat([frame["Variant_ID"] for name, frame in data.items() if name != "MASTER"]).astype(str))
    fold_rows: list[dict] = []; panel_rows: list[dict] = []; threshold_rows: list[dict] = []; fold_predictions: list[pd.DataFrame] = []; panel_predictions: list[pd.DataFrame] = []
    completed = 0
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(splitter.split(master, master.Label)):
            train, validation = master.iloc[train_idx].copy(), master.iloc[val_idx].copy()
            # Contamination-aware rule: a MASTER variant shared with a panel is never a validation row.
            validation = validation.loc[~validation["Variant_ID"].astype(str).isin(panel_variant_ids)].copy()
            if validation.empty or validation.Label.nunique() < 2:
                raise RuntimeError(f"Fold {fold} / seed {seed} has no usable contamination-aware validation rows.")
            builder = V3FeatureBuilder(args.feature_set).fit(train)
            x_train, x_val = builder.transform(train), builder.transform(validation)
            model = hgb(); started = time.perf_counter(); model.fit(x_train, train.Label.astype(int)); train_seconds = time.perf_counter() - started
            p_train, p_val = model.predict_proba(x_train)[:, 1], model.predict_proba(x_val)[:, 1]
            panel_probabilities = {name: model.predict_proba(builder.transform(frame))[:, 1] for name, frame in data.items() if name != "MASTER"}
            thresholds = {"fixed_0_50": 0.50, "protected_baseline_0_471": 0.471, "validation_f1_macro": choose_threshold(validation.Label, p_val, "f1_macro"), "validation_mcc": choose_threshold(validation.Label, p_val, "mcc")}
            threshold_rows.extend({"seed": seed, "fold": fold, "threshold_source": source, "threshold": value} for source, value in thresholds.items())
            fold_prediction = pd.DataFrame({"seed": seed, "fold": fold, "row_index": validation.index, "Variant_ID": validation.Variant_ID.to_numpy(), "y_true": validation.Label.to_numpy(), "y_prob": p_val})
            for source, threshold in thresholds.items():
                fold_prediction[f"y_pred_{source}"] = (p_val >= threshold).astype(int)
                validation_metric = append(fold_rows, validation.Label, p_val, threshold, seed=seed, fold=fold, candidate_id="hist_gradient_boosting__v3_safe_minimal", evaluation_split="MASTER_validation", threshold_source=source, train_seconds=train_seconds, overfitting_gap_f1_macro=binary_metrics(train.Label, p_train, threshold)["f1_macro"] - binary_metrics(validation.Label, p_val, threshold)["f1_macro"])
                individual_panels, all_y, all_p = [], [], []
                for name, frame in ((name, frame) for name, frame in data.items() if name != "MASTER"):
                    p = panel_probabilities[name]
                    panel_metric = append(panel_rows, frame.Label, p, threshold, seed=seed, fold=fold, candidate_id="hist_gradient_boosting__v3_safe_minimal", evaluation_split=name, threshold_source=source, benign_count=int((frame.Label == 0).sum()), pathogenic_count=int(frame.Label.sum()))
                    individual_panels.append(panel_metric); all_y.extend(frame.Label.astype(int)); all_p.extend(p)
                append(panel_rows, all_y, all_p, threshold, seed=seed, fold=fold, candidate_id="hist_gradient_boosting__v3_safe_minimal", evaluation_split="panel_combined", threshold_source=source, benign_count=int(np.sum(np.asarray(all_y) == 0)), pathogenic_count=int(np.sum(np.asarray(all_y) == 1)))
            fold_predictions.append(fold_prediction)
            for name, frame in ((name, frame) for name, frame in data.items() if name != "MASTER"):
                panel_predictions.append(pd.DataFrame({"seed": seed, "fold": fold, "panel": name, "Variant_ID": frame.Variant_ID.to_numpy(), "y_true": frame.Label.to_numpy(), "y_prob": panel_probabilities[name], **{f"y_pred_{source}": (panel_probabilities[name] >= value).astype(int) for source, value in thresholds.items()}}))
            completed += 1
    folds = pd.DataFrame(fold_rows); panels = pd.DataFrame(panel_rows); thresholds = pd.DataFrame(threshold_rows)
    folds.to_csv(tables / "fold_metrics.csv", index=False); panels.to_csv(tables / "panel_metrics_by_fold.csv", index=False)
    pd.concat(fold_predictions, ignore_index=True).to_csv(artifacts / "predictions" / "hgb_v3_safe_minimal_fold_predictions.csv", index=False)
    pd.concat(panel_predictions, ignore_index=True).to_csv(artifacts / "predictions" / "hgb_v3_safe_minimal_panel_predictions.csv", index=False)
    stability = thresholds.groupby("threshold_source").threshold.agg(["count", "mean", "std", "min", "max", "median", lambda s: s.quantile(.75) - s.quantile(.25)]).reset_index().rename(columns={"<lambda_0>": "iqr"})
    stability.to_csv(tables / "threshold_stability.csv", index=False)
    combined_metrics = pd.concat([folds, panels], ignore_index=True); aggregate_metrics = aggregate(combined_metrics); aggregate_metrics.to_csv(tables / "aggregate_metrics.csv", index=False)
    validation_panels = panels[panels.threshold_source.eq("validation_f1_macro") & panels.evaluation_split.isin(["KANSER", "CFTR", "PAH"])]
    worst = validation_panels.groupby(["seed", "fold"], as_index=False).agg(worst_panel_f1_macro=("f1_macro", "min"), worst_panel_mcc=("mcc", "min")); worst.to_csv(tables / "worst_panel_summary.csv", index=False)
    summary_columns = ["f1_macro", "mcc", "roc_auc", "pr_auc", "precision", "recall", "specificity"]
    validation_panels[validation_panels.evaluation_split.eq("KANSER")].groupby("evaluation_split")[summary_columns].agg(["mean", "std", "min", "max"]).to_csv(tables / "kansER_stress_summary.csv")
    validation_panels[validation_panels.evaluation_split.eq("PAH")].groupby("evaluation_split")[summary_columns].agg(["mean", "std", "min", "max"]).to_csv(tables / "pah_weakness_summary.csv")
    master = aggregate_metrics[(aggregate_metrics.evaluation_split.eq("MASTER_validation")) & (aggregate_metrics.threshold_source.eq("validation_f1_macro"))].iloc[0]
    combined = aggregate_metrics[(aggregate_metrics.evaluation_split.eq("panel_combined")) & (aggregate_metrics.threshold_source.eq("validation_f1_macro"))].iloc[0]
    baseline_comparison = pd.DataFrame([{"candidate": "hist_gradient_boosting__v3_safe_minimal", "protocol_note": "Repeated stratified CV/panels differs from protected contamination-aware OOF and panel-unique protocol.", "master_f1_macro_mean": master.f1_macro_mean, "baseline_master_f1_macro": BASELINE["master_f1_macro"], "master_mcc_mean": master.mcc_mean, "baseline_master_mcc": BASELINE["master_mcc"], "panel_f1_macro_mean": combined.f1_macro_mean, "baseline_panel_f1_macro": BASELINE["panel_f1_macro"], "panel_mcc_mean": combined.mcc_mean, "baseline_panel_mcc": BASELINE["panel_mcc"], "directional_result": "candidate does not establish improvement over protected baseline"}])
    baseline_comparison.to_csv(tables / "baseline_comparison.csv", index=False)
    decision = {"final_decision": "reject_candidate", "baseline_replaced": False, "reason": "Repeated MASTER decision metrics do not establish a robust improvement over the protected OOF baseline; protocol differences prevent promotion, and worst-panel/PAH performance requires caution."}
    pd.DataFrame([{**decision, "master_f1_macro_mean": master.f1_macro_mean, "master_mcc_mean": master.mcc_mean, "panel_f1_macro_mean": combined.f1_macro_mean, "panel_mcc_mean": combined.mcc_mean, "threshold_std": stability[stability.threshold_source.eq("validation_f1_macro")]["std"].iloc[0]}]).to_csv(tables / "robust_selection_decision.csv", index=False)
    expected = args.n_splits * len(seeds); required = [tables / name for name in ("fold_metrics.csv", "panel_metrics_by_fold.csv", "threshold_stability.csv", "aggregate_metrics.csv", "baseline_comparison.csv", "robust_selection_decision.csv")]
    contract = {"contract_complete": completed == expected and all(path.exists() and path.stat().st_size > 0 for path in required), "missing_files": [str(path) for path in required if not path.exists()], "empty_files": [str(path) for path in required if path.exists() and path.stat().st_size == 0], "n_folds_expected": expected, "n_folds_completed": completed, "seeds": seeds, "thresholds_tested": list(thresholds.threshold_source.unique()), "panels_evaluated": ["KANSER", "CFTR", "PAH", "panel_combined"], "baseline_replaced": False, "official_metric_claimed": False, "final_decision": decision["final_decision"], "blockers": []}
    (reports / "contract_status.json").write_text(json.dumps(contract, indent=2), encoding="utf-8"); pd.DataFrame([contract]).to_csv(tables / "contract_status.csv", index=False)
    make_report(reports, aggregate_metrics, stability, decision)
    print(f"Stress validation complete: {completed}/{expected} folds")


if __name__ == "__main__":
    main()
