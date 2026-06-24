"""Controlled, leakage-safe local-holdout evaluation for V3 candidates.

This never trains on local_test or panels, never uses an identifier as a model
feature, and does not make a final-model decision.  It writes only below the
isolated artifacts/v3 and reports/v3 namespaces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from teknofest_v3.data import DATASETS, find_unlabeled_test_files, load_labeled_competition_data
from teknofest_v3.features import V3FeatureBuilder
from teknofest_v3.metrics import binary_metrics, choose_threshold
from teknofest_v3.selection import robust_genomics_score


def make_model(family: str, quick: bool):
    """Fixed, conservative configurations only; no tuning or Optuna."""
    if family == "logistic_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=42)),
        ])
    if family == "extratrees":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesClassifier(n_estimators=120 if quick else 300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=4))])
    if family == "hist_gradient_boosting":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=120 if quick else 220, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=3.0, random_state=42))])
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=150 if quick else 350, learning_rate=0.035, num_leaves=24, max_depth=5, min_child_samples=60, reg_alpha=1.0, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.75, class_weight="balanced", random_state=42, n_jobs=4, verbosity=-1)
    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=120 if quick else 250, max_depth=4, learning_rate=0.04, min_child_weight=5, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=5.0, scale_pos_weight=0.36, objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=4)
    if family == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=120 if quick else 250, depth=5, learning_rate=0.04, l2_leaf_reg=6.0, loss_function="Logloss", verbose=False, random_seed=42, thread_count=4, auto_class_weights="Balanced")
    raise ValueError(f"Unknown model family: {family}")


def proba(model, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def append_metric(rows: list[dict], y, probability, threshold: float, **metadata) -> dict:
    row = binary_metrics(y, probability, threshold)
    row.update(metadata)
    rows.append(row)
    return row


def model_availability(families: list[str], quick: bool) -> tuple[list[str], list[dict]]:
    available, rows = [], []
    for family in families:
        try:
            make_model(family, quick)
            available.append(family)
            rows.append({"model_family": family, "available": True, "status": "available", "reason": ""})
        except Exception as exc:
            rows.append({"model_family": family, "available": False, "status": "skipped", "reason": f"{type(exc).__name__}: {exc}"})
    return available, rows


def prediction_frame(frame: pd.DataFrame, probability: np.ndarray, thresholds: dict[str, float], model_id: str, feature_set: str, dataset: str) -> pd.DataFrame:
    out = pd.DataFrame({"row_index": frame.index.to_numpy(), "Variant_ID": frame["Variant_ID"].to_numpy(), "y_true": frame["Label"].astype(int).to_numpy(), "y_prob": probability, "dataset": dataset, "model_id": model_id, "feature_set": feature_set})
    out["y_pred_0_50"] = (probability >= 0.50).astype(int)
    out["y_pred_0_471"] = (probability >= 0.471).astype(int)
    for source, threshold in thresholds.items():
        out[f"y_pred_{source}"] = (probability >= threshold).astype(int)
    return out


def write_report(reports: Path, split: pd.DataFrame, features: pd.DataFrame, availability: pd.DataFrame, warnings_df: pd.DataFrame, metrics: pd.DataFrame, panels: pd.DataFrame, selection: pd.DataFrame, official: pd.DataFrame) -> None:
    local = metrics[metrics["evaluation_split"].eq("local_test")]
    best_f1 = local.sort_values(["f1_macro", "mcc"], ascending=False).iloc[0]
    best_mcc = local.sort_values(["mcc", "f1_macro"], ascending=False).iloc[0]
    best_selection = selection.sort_values("robust_genomics_score", ascending=False).iloc[0]
    best_panels = panels[(panels["candidate_id"].eq(best_selection["candidate_id"])) & (panels["threshold_source"].eq(best_selection["threshold_source"]))]
    combined = best_panels[best_panels["evaluation_split"].eq("panel_combined")].iloc[0]
    individual = best_panels[best_panels["evaluation_split"].isin(["KANSER", "CFTR", "PAH"])]
    equivalence = features[["feature_set", "feature_count", "equivalent_to"]].to_markdown(index=False)
    text = f"""# Controlled V3 Local-Holdout Training and Evaluation

## Executive summary

This is an **internal local-holdout evaluation**, not official hidden-test performance. All learned feature fitting, imputation, scaling, and model fitting used MASTER `train` rows only. Thresholds were selected on `validation` only; `local_test` was not used for feature fitting, threshold selection, or candidate promotion. The protected baseline remains final.

## Data files and official-test status

Labeled competition training files: {', '.join(DATASETS.values())}.

{official.to_markdown(index=False)}

No official test metric is claimed.

## Split and class distribution

{split.to_markdown(index=False)}

The requested 80/20 MASTER split was created first; its 80% training side was split into train/validation (80/20) solely for validation-derived thresholds.

## Feature sets

{equivalence}

All outputs are numeric; `Variant_ID` and `Label` are excluded from features. An equivalence entry identifies feature matrices that are exactly identical rather than presenting them as independent evidence.

## Model-family handling and warnings

{availability.to_markdown(index=False)}

Logistic Regression uses `SimpleImputer → StandardScaler → LogisticRegression(lbfgs, max_iter=5000, class_weight=balanced)`. Captured warnings:

{warnings_df.to_markdown(index=False) if not warnings_df.empty else 'No training warnings captured.'}

## Best internal results

Best local-holdout F1-macro: `{best_f1['candidate_id']}` at `{best_f1['threshold_source']}` — F1-macro {best_f1['f1_macro']:.4f}, MCC {best_f1['mcc']:.4f}.

Best local-holdout MCC: `{best_mcc['candidate_id']}` at `{best_mcc['threshold_source']}` — F1-macro {best_mcc['f1_macro']:.4f}, MCC {best_mcc['mcc']:.4f}.

Best exploratory robust-selection candidate: `{best_selection['candidate_id']}` at `{best_selection['threshold_source']}` — score {best_selection['robust_genomics_score']:.4f}. This is **not** a final-selection result.

## Panel results, KANSER, and worst panel

{individual[['evaluation_split','n','f1_macro','mcc','roc_auc','pr_auc','precision','recall','specificity','tn','fp','fn','tp']].to_markdown(index=False)}

Combined panel: F1-macro {combined['f1_macro']:.4f}; MCC {combined['mcc']:.4f}; PR-AUC {combined['pr_auc']:.4f}. Worst-panel values are F1-macro {best_selection['worst_panel_f1_macro']:.4f} and MCC {best_selection['worst_panel_mcc']:.4f}.

## Confusion-matrix and threshold interpretation

The best robust candidate's local-holdout confusion matrix is TN {int(best_selection['tn'])}, FP {int(best_selection['fp'])}, FN {int(best_selection['fn'])}, TP {int(best_selection['tp'])}. Threshold variants (0.50, protected 0.471, validation F1-macro, validation MCC) are recorded without optimizing on local_test.

## Robust comparison and final decision

Every candidate is rejected from final replacement. The protected baseline has MASTER OOF F1-macro 0.7764 and MCC 0.5548 with panel-unique F1-macro 0.7708 and MCC 0.5825. Internal holdout performance is not protocol-comparable, and no candidate can pass the robust replacement gates from this experiment. **Protected baseline remains final.**

## Limitations and next action

- Internal evaluation is not official hidden-test performance.
- Panel training files are labeled external checks, but their sizes and disease distributions differ from MASTER.
- No official test CSV was found locally; no submission predictions were produced.
- Do not run Optuna from this phase. The next valid action, if requested, is a repeated contamination-aware V3 validation of one candidate under the protected baseline's protocol.

## Reproduce

`python scripts/v3_train_test_evaluate.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main --output-dir reports/v3/train_test --test-size 0.20 --random-state 42 --feature-sets v3_safe_minimal,v3_no_target_encoding,v3_frequency_heavy,v3_panel_robust --models logistic_regression,extratrees,hist_gradient_boosting,lightgbm,xgboost,catboost --quick`
"""
    (reports / "train_test_evaluation_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="reports/v3/train_test")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--feature-sets", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not 0.05 <= args.test_size < 0.5:
        raise ValueError("--test-size must be between 0.05 and 0.50")

    reports = ROOT / args.output_dir
    tables, artifacts = reports / "tables", ROOT / "artifacts" / "v3" / "train_test"
    for folder in (tables, artifacts / "models", artifacts / "predictions", artifacts / "features", artifacts / "schemas"):
        folder.mkdir(parents=True, exist_ok=True)
    data = load_labeled_competition_data(args.data_dir)
    master = data["MASTER"].reset_index(drop=True)
    train_valid, local_test = train_test_split(master, test_size=args.test_size, stratify=master.Label, random_state=args.random_state)
    train, validation = train_test_split(train_valid, test_size=0.20, stratify=train_valid.Label, random_state=args.random_state)
    split = pd.DataFrame([{"split": name, "n": len(frame), "benign": int((frame.Label == 0).sum()), "pathogenic": int(frame.Label.sum()), "pathogenic_rate": float(frame.Label.mean())} for name, frame in (("train", train), ("validation", validation), ("local_test", local_test), *data.items())])
    split.to_csv(tables / "train_test_split_summary.csv", index=False)
    requested_families = [x.strip() for x in args.models.split(",") if x.strip()]
    families, availability_rows = model_availability(requested_families, args.quick)
    availability = pd.DataFrame(availability_rows); availability.to_csv(tables / "model_availability.csv", index=False)
    feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]
    metric_rows: list[dict] = []; panel_rows: list[dict] = []; confusion_rows: list[dict] = []; panel_confusion_rows: list[dict] = []; selection_rows: list[dict] = []; warning_rows: list[dict] = []; feature_rows: list[dict] = []
    feature_matrices: dict[str, tuple[V3FeatureBuilder, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]] = {}
    for feature_set in feature_sets:
        builder = V3FeatureBuilder(feature_set).fit(train)
        x_train, x_validation, x_test = builder.transform(train), builder.transform(validation), builder.transform(local_test)
        panels_x = {name: builder.transform(frame) for name, frame in data.items() if name != "MASTER"}
        matrix_hash = hashlib.sha256(pd.util.hash_pandas_object(x_train, index=True).values.tobytes()).hexdigest()
        equivalent = next((row["feature_set"] for row in feature_rows if row["matrix_hash"] == matrix_hash), "")
        feature_rows.append({"feature_set": feature_set, "feature_count": len(builder.feature_names_), "matrix_hash": matrix_hash, "equivalent_to": equivalent or "none", "contains_variant_id": False, "contains_label": False, "numeric_only": True})
        builder.save_schema(artifacts / "schemas" / f"{feature_set}.json")
        pd.DataFrame({"feature_name": builder.feature_names_}).to_csv(artifacts / "features" / f"{feature_set}_features.csv", index=False)
        feature_matrices[feature_set] = (builder, x_train, x_validation, x_test, panels_x)
    features = pd.DataFrame(feature_rows); features.to_csv(tables / "feature_set_comparison.csv", index=False)
    for feature_set, (builder, x_train, x_validation, x_test, panels_x) in feature_matrices.items():
        for model_id in families:
            candidate_id = f"{model_id}__{feature_set}"
            model = make_model(model_id, args.quick)
            try:
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(x_train, train.Label.astype(int))
                train_seconds = time.perf_counter() - started
                for warning in caught:
                    warning_rows.append({"candidate_id": candidate_id, "model_id": model_id, "feature_set": feature_set, "warning_category": warning.category.__name__, "message": str(warning.message), "is_convergence_warning": issubclass(warning.category, ConvergenceWarning)})
                started = time.perf_counter()
                p_train, p_valid, p_test = proba(model, x_train), proba(model, x_validation), proba(model, x_test)
                p_panels = {name: proba(model, matrix) for name, matrix in panels_x.items()}
                prediction_seconds = time.perf_counter() - started
            except Exception as exc:
                availability.loc[availability.model_family.eq(model_id), ["status", "reason"]] = ["training_failed", f"{type(exc).__name__}: {exc}"]
                continue
            thresholds = {"fixed_0_50": 0.50, "protected_baseline_0_471": 0.471, "validation_f1_macro": choose_threshold(validation.Label, p_valid, "f1_macro"), "validation_mcc": choose_threshold(validation.Label, p_valid, "mcc")}
            joblib.dump(model, artifacts / "models" / f"{candidate_id}.joblib")
            joblib.dump(builder, artifacts / "models" / f"{candidate_id}_feature_builder.joblib")
            (artifacts / "models" / f"{candidate_id}_feature_names.json").write_text(json.dumps(builder.feature_names_, indent=2), encoding="utf-8")
            (artifacts / "models" / f"{candidate_id}_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
            (artifacts / "models" / f"{candidate_id}_metadata.json").write_text(json.dumps({"model_id": model_id, "feature_set": feature_set, "feature_count": len(builder.feature_names_), "trained_on": "MASTER train only", "random_state": args.random_state}, indent=2), encoding="utf-8")
            prediction_frame(local_test, p_test, thresholds, model_id, feature_set, "local_test").to_csv(artifacts / "predictions" / f"{candidate_id}_local_test_predictions.csv", index=False)
            combined_predictions = []
            for name, frame in ((name, frame) for name, frame in data.items() if name != "MASTER"):
                combined_predictions.append(prediction_frame(frame, p_panels[name], thresholds, model_id, feature_set, name))
            pd.concat(combined_predictions, ignore_index=True).to_csv(artifacts / "predictions" / f"{candidate_id}_panel_predictions.csv", index=False)
            for source, threshold in thresholds.items():
                local = append_metric(metric_rows, local_test.Label, p_test, threshold, candidate_id=candidate_id, model_id=model_id, feature_set=feature_set, evaluation_split="local_test", threshold_source=source, train_seconds=train_seconds, prediction_seconds=prediction_seconds, feature_count=len(builder.feature_names_), overfitting_gap_f1_macro=binary_metrics(train.Label, p_train, threshold)["f1_macro"] - binary_metrics(local_test.Label, p_test, threshold)["f1_macro"])
                confusion_rows.append({"candidate_id": candidate_id, "model_id": model_id, "feature_set": feature_set, "threshold": threshold, "threshold_source": source, "tn": local["tn"], "fp": local["fp"], "fn": local["fn"], "tp": local["tp"], "false_positive_rate": local["fp"] / (local["fp"] + local["tn"]), "false_negative_rate": local["fn"] / (local["fn"] + local["tp"]), "specificity": local["specificity"], "recall": local["recall"], "precision": local["precision"], "f1_macro": local["f1_macro"], "mcc": local["mcc"]})
                one_threshold_panels = []
                all_y, all_p = [], []
                for panel_name, panel in ((name, frame) for name, frame in data.items() if name != "MASTER"):
                    panel_metric = append_metric(panel_rows, panel.Label, p_panels[panel_name], threshold, candidate_id=candidate_id, model_id=model_id, feature_set=feature_set, evaluation_split=panel_name, threshold_source=source)
                    one_threshold_panels.append(panel_metric); all_y.extend(panel.Label.astype(int)); all_p.extend(p_panels[panel_name])
                    panel_confusion_rows.append({"candidate_id": candidate_id, "model_id": model_id, "feature_set": feature_set, "panel": panel_name, "threshold": threshold, "threshold_source": source, "tn": panel_metric["tn"], "fp": panel_metric["fp"], "fn": panel_metric["fn"], "tp": panel_metric["tp"], "false_positive_rate": panel_metric["fp"] / (panel_metric["fp"] + panel_metric["tn"]), "false_negative_rate": panel_metric["fn"] / (panel_metric["fn"] + panel_metric["tp"]), "specificity": panel_metric["specificity"], "recall": panel_metric["recall"], "precision": panel_metric["precision"], "f1_macro": panel_metric["f1_macro"], "mcc": panel_metric["mcc"]})
                combined = append_metric(panel_rows, all_y, all_p, threshold, candidate_id=candidate_id, model_id=model_id, feature_set=feature_set, evaluation_split="panel_combined", threshold_source=source)
                selection_rows.append({"candidate_id": candidate_id, "model_id": model_id, "feature_set": feature_set, "threshold": threshold, "threshold_source": source, **{f"local_test_{key}": local[key] for key in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1_macro", "mcc", "roc_auc", "pr_auc")}, "panel_combined_f1_macro": combined["f1_macro"], "panel_combined_mcc": combined["mcc"], "panel_combined_pr_auc": combined["pr_auc"], **{f"{name}_{key}": next(row[key] for row in one_threshold_panels if row["evaluation_split"] == name) for name in ("KANSER", "CFTR", "PAH") for key in ("f1_macro", "mcc")}, "worst_panel_f1_macro": min(row["f1_macro"] for row in one_threshold_panels), "worst_panel_mcc": min(row["mcc"] for row in one_threshold_panels), "robust_genomics_score": robust_genomics_score(local, one_threshold_panels), "baseline_robust_genomics_score": np.nan, "replacement_allowed": False, "rejection_reasons": "internal holdout protocol is not comparable to protected contamination-aware OOF; local F1-macro/MCC cannot establish a robust improvement; no final replacement permitted", "selected_as_candidate": False, "selected_as_final": False, "tn": local["tn"], "fp": local["fp"], "fn": local["fn"], "tp": local["tp"]})
    availability.to_csv(tables / "model_availability.csv", index=False)
    warnings_df = pd.DataFrame(warning_rows, columns=["candidate_id", "model_id", "feature_set", "warning_category", "message", "is_convergence_warning"]); warnings_df.to_csv(tables / "model_warnings.csv", index=False)
    metrics, panels, selection = pd.DataFrame(metric_rows), pd.DataFrame(panel_rows), pd.DataFrame(selection_rows)
    metrics.to_csv(tables / "model_train_test_metrics.csv", index=False); panels.to_csv(tables / "model_panel_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(tables / "model_confusion_matrices.csv", index=False); pd.DataFrame(panel_confusion_rows).to_csv(tables / "model_panel_confusion_matrices.csv", index=False)
    metrics.to_csv(tables / "model_threshold_comparison.csv", index=False)
    if not selection.empty:
        winner = selection["robust_genomics_score"].idxmax(); selection.loc[winner, "selected_as_candidate"] = True
    selection.to_csv(tables / "model_selection_comparison.csv", index=False)
    test_files = find_unlabeled_test_files(args.data_dir)
    official = pd.DataFrame([{"file_path": str(path) if test_files else "none", "exists": bool(test_files), "has_label_column": False, "action_taken": "no official test CSV found" if not test_files else "no predictions generated because no candidate is final", "metrics_computed": False, "prediction_file_created": False, "notes": "no official test metric is claimed"}])
    official.to_csv(tables / "official_test_prediction_status.csv", index=False)
    write_report(reports, split, features, availability, warnings_df, metrics, panels, selection, official)
    print(f"Controlled local-holdout contract complete: {len(selection)} candidate-threshold rows; reports in {reports}")


if __name__ == "__main__":
    main()
