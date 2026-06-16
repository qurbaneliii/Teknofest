from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

from config import ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, PREPROCESSORS_DIR, TABLES_DIR
from baselines import run_baselines
from data_loading import discover_data_dir, write_data_diagnostics
from evaluation import save_evaluation_figures, threshold_results
from phase9_outputs import generate_phase9_outputs
from teknofest.data_prep import prepare_data
from teknofest.experiments import final_panel_predictions, panel_bootstrap_reports, run_ablation_table
from teknofest.features import FeatureEngineer, detect_binary_al_cols
from teknofest.training import fit_final_lgbm
from teknofest.validation import contamination_aware_folds, fold_summary


def load_params() -> dict[str, object]:
    for path in [
        Path("reports/master_prompt/lgbm_best_params_resumable.json"),
        Path("data/processed/lgbm_best_params.json"),
    ]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "scale_pos_weight": 0.35,
    }


def ensure_dirs() -> None:
    for path in [TABLES_DIR, FIGURES_DIR, MODELS_DIR, PREPROCESSORS_DIR, PREDICTIONS_DIR, METRICS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def save_feature_list(prepared) -> None:
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
    engineered = engineer.fit_transform(prepared.master)
    feature_cols = [c for c in engineered.columns if c not in {"Variant_ID", "Label"}]
    (METRICS_DIR / "feature_list.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")


def save_validation_diagnostics(prepared) -> None:
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    fold_summary(prepared.master, folds).to_csv(TABLES_DIR / "validation_split_diagnostics.csv", index=False)


def save_baselines(prepared) -> None:
    baseline = run_baselines(prepared)
    baseline.to_csv(TABLES_DIR / "baseline_results.csv", index=False)


def save_main_outputs(prepared, mode: str) -> None:
    params = load_params()
    source_oof = Path("reports/master_prompt/lightgbm_oof_predictions.csv")
    if not copy_if_exists(source_oof, PREDICTIONS_DIR / "oof_predictions.csv"):
        run_ablation_table(prepared, params, Path("reports/master_prompt"), n_estimators=20 if mode == "smoke" else 350)
        copy_if_exists(source_oof, PREDICTIONS_DIR / "oof_predictions.csv")

    oof = pd.read_csv(PREDICTIONS_DIR / "oof_predictions.csv")
    y = oof["Label"]
    score = oof["lightgbm_probability"]
    threshold_results(y, score).to_csv(TABLES_DIR / "threshold_results.csv", index=False)
    save_evaluation_figures(y, score, FIGURES_DIR)

    thr = pd.read_csv(TABLES_DIR / "threshold_results.csv")
    plt.figure(figsize=(7, 4))
    plt.bar(thr["threshold_name"], thr["f1_macro"], color="#315f72")
    plt.ylabel("F1-macro")
    plt.title("Threshold comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_comparison.png", dpi=170)
    plt.close()

    oof_metrics = pd.read_csv("reports/master_prompt/ablation_summary.csv") if Path("reports/master_prompt/ablation_summary.csv").exists() else pd.DataFrame()
    if not oof_metrics.empty:
        oof_metrics.to_csv(TABLES_DIR / "main_model_cv_results.csv", index=False)
        plt.figure(figsize=(9, 5))
        subset = oof_metrics[oof_metrics["threshold_name"].isin(["f1_macro_opt", "extra_trees_default_0.5", "comparison"])]
        plt.barh(subset["ablation"], subset["CV_AUC"], color="#476f5d")
        plt.xlabel("CV AUC")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=170)
        plt.close()

    source_panel = Path("reports/master_prompt/final_panel_unique_predictions.csv")
    if not copy_if_exists(source_panel, PREDICTIONS_DIR / "panel_unique_predictions.csv"):
        engineer, model, cols = fit_final_lgbm(prepared, params, MODELS_DIR)
        panel = final_panel_predictions(prepared, params, MODELS_DIR, Path("reports/master_prompt"), 0.5)
        panel.to_csv(PREDICTIONS_DIR / "panel_unique_predictions.csv", index=False)

    panel = pd.read_csv(PREDICTIONS_DIR / "panel_unique_predictions.csv")
    panel_rows = []
    errors = []
    for dataset, group in panel.groupby("dataset"):
        acc = (group["Label"] == group["predicted_label"]).mean()
        panel_rows.append({"dataset": dataset, "n": len(group), "accuracy_at_saved_threshold": float(acc)})
        errors.append(group[group["Label"] != group["predicted_label"]].assign(error_type=lambda d: d["Label"].map({0: "FP", 1: "FN"})))
    pd.DataFrame(panel_rows).to_csv(TABLES_DIR / "panel_generalization_results.csv", index=False)
    pd.concat(errors, ignore_index=True).to_csv(TABLES_DIR / "error_analysis.csv", index=False)
    panel_bootstrap_reports(panel, TABLES_DIR, n_bootstrap=200 if mode == "smoke" else 1000)
    mirror_model_artifacts()


def mirror_model_artifacts() -> None:
    copy_if_exists(Path("models/lightgbm_final.joblib"), MODELS_DIR / "lightgbm_final.joblib")
    copy_if_exists(Path("models/model_columns.txt"), MODELS_DIR / "model_columns.txt")
    copy_if_exists(Path("models/feature_engineer.joblib"), PREPROCESSORS_DIR / "feature_engineer.joblib")


def save_explainability_exports() -> None:
    copy_if_exists(Path("reports/explainability/shap_global_importance.csv"), TABLES_DIR / "feature_importance.csv")
    copy_if_exists(Path("reports/explainability/acmg_feature_mapping.csv"), TABLES_DIR / "acmg_feature_mapping.csv")
    copy_if_exists(Path("reports/explainability/shap_global_bar.png"), FIGURES_DIR / "feature_importance.png")


def save_final_report_summary(prepared) -> None:
    optuna_count = "not available"
    optuna_best = "not available"
    trials_path = Path("reports/master_prompt/lgbm_optuna_trials_resumable.csv")
    if trials_path.exists():
        trials = pd.read_csv(trials_path)
        complete = trials[trials["state"] == "COMPLETE"]
        optuna_count = str(len(complete))
        if not complete.empty:
            optuna_best = f"{complete['mean_auc'].max():.5f}"

    text = f"""# Final Model Report Summary

## Problem Definition
Binary classification of clinical genomics missense variants as Pathogenic or Benign for TEKNOFEST 2026 Healthcare AI, University level.

## Dataset Summary
MASTER has {len(prepared.master)} variants. KANSER, PAH, and CFTR contain {len(prepared.kanser)}, {len(prepared.pah)}, and {len(prepared.cftr)} variants. MASTER-shared variants are excluded from validation folds.

## Method Summary
The pipeline uses leakage-safe feature engineering, contamination-aware cross-validation, LightGBM-centered tabular modeling, panel-unique external checks, and SHAP explainability.

## Feature Engineering Summary
Features include missingness indicators/PCA, population allele-frequency aggregates, ACMG-inspired BA1/BS1/PM2/BS2 flags, EK interactions/evidence counts, CAT decompositions, and AA physicochemical features.

## Validation Strategy
Primary validation is 5-fold StratifiedKFold on MASTER with MASTER-shared variants removed from validation folds. Secondary validation evaluates KANSER-unique, PAH-unique, and CFTR-unique subsets.

## Model Architecture
Baselines include majority class, ACMG rule engine, and EK-only logistic regression. Main models include LightGBM plus optional CatBoost/XGBoost/ExtraTrees/LR stack. The final LightGBM parameters come from the resumable Optuna study with {optuna_count} complete trials and best contamination-aware CV AUC of {optuna_best}.

## Metrics Table References
See `reports/tables/all_evaluation_metrics.csv`, `baseline_results.csv`, `main_model_cv_results.csv`, `panel_generalization_results.csv`, and `threshold_results.csv`.

## Required Visualization References
Phase 9.5 visualizations are saved under `reports/figures`: `correlation_matrix_top_features.png`, confusion matrices for MASTER and panel-unique splits, `roc_curve_master.png`, `roc_curve_panel_unique.png`, `pr_curve_master.png`, `pr_curve_panel_unique.png`, `threshold_optimization.png`, `model_comparison_metrics.png`, `feature_importance_top30.png`, `class_distribution_by_dataset.png`, `missingness_by_feature_group.png`, and `error_analysis_key_features.png`.

## Explainability Summary
See `reports/tables/feature_importance.csv`, `acmg_feature_mapping.csv`, and `reports/figures/feature_importance.png`.

## Error Analysis Summary
Panel-unique false positives and false negatives are saved in `reports/tables/error_analysis.csv`.

## Limitations And Next Steps
Final competition performance depends on the hidden external validation set distribution.
"""
    Path("reports/final_model_report_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TEKNOFEST 2026 end-to-end pipeline.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    ensure_dirs()
    data_dir = discover_data_dir(Path.cwd())
    prepared = prepare_data(data_dir)
    write_data_diagnostics(data_dir, TABLES_DIR)
    save_validation_diagnostics(prepared)
    save_feature_list(prepared)
    save_baselines(prepared)
    save_main_outputs(prepared, args.mode)
    save_explainability_exports()
    generate_phase9_outputs(prepared)
    save_final_report_summary(prepared)
    print(f"Pipeline complete in {args.mode} mode.")
    print(f"Tables: {TABLES_DIR.resolve()}")
    print(f"Figures: {FIGURES_DIR.resolve()}")
    print(f"Artifacts: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
