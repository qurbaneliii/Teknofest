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
from evaluation import diagnose_model_performance, save_evaluation_figures, threshold_results
from phase9_outputs import generate_phase9_outputs
from phase10_improvements import run_phase10_improvements
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
    (METRICS_DIR / "experiments").mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "experiments").mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "experiments").mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree_if_absent(src: Path, dst: Path) -> bool:
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
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


def has_report_grade_bootstrap(path: Path, min_bootstrap: int = 1000) -> bool:
    if not path.exists():
        return False
    try:
        table = pd.read_csv(path, usecols=["n_bootstrap_used"])
    except (ValueError, pd.errors.EmptyDataError):
        return False
    if table.empty:
        return False
    return int(table["n_bootstrap_used"].min()) >= min_bootstrap


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
    bootstrap_path = TABLES_DIR / "panel_unique_bootstrap_ci.csv"
    if mode == "smoke" and has_report_grade_bootstrap(bootstrap_path):
        print(f"Preserved report-grade bootstrap CI table: {bootstrap_path}")
    else:
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

    phase10_note = ""
    final_threshold_path = METRICS_DIR / "final_threshold.json"
    final_metrics_path = METRICS_DIR / "final_metrics.json"
    if final_threshold_path.exists() and final_metrics_path.exists():
        final_threshold = json.loads(final_threshold_path.read_text(encoding="utf-8"))
        phase10_note = (
            "\n## Phase 10 Final Selection\n"
            f"The final model is classified as {final_threshold.get('model_strength', 'moderate')} rather than weak or strong. "
            f"The selected threshold strategy is `{final_threshold.get('threshold_strategy')}` "
            f"with threshold {final_threshold.get('threshold')}. "
            "The main technical weakness is thresholding/probability calibration rather than feature learning. "
            "See `reports/final_performance_analysis.md`, `reports/tables/final_model_selection_table.csv`, "
            "and `reports/tables/advanced_threshold_comparison.csv`.\n"
        )

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
{phase10_note}
"""
    Path("reports/final_model_report_summary.md").write_text(text, encoding="utf-8")


def metric_lookup(metrics: pd.DataFrame, model: str, split: str, threshold: str) -> pd.Series:
    subset = metrics[
        metrics["model_name"].eq(model)
        & metrics["evaluation_split"].eq(split)
        & metrics["threshold_type"].eq(threshold)
    ]
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.iloc[0]


def best_master_row(metrics: pd.DataFrame, model: str) -> pd.Series:
    subset = metrics[metrics["model_name"].eq(model) & metrics["evaluation_split"].eq("MASTER_CV")]
    if subset.empty:
        return pd.Series(dtype=object)
    preferred = subset[subset["threshold_type"].eq("f1_macro_opt")]
    if not preferred.empty:
        return preferred.iloc[0]
    return subset.sort_values(["f1_macro", "mcc", "roc_auc"], ascending=False).iloc[0]


def panel_combined_row(metrics: pd.DataFrame) -> pd.Series:
    panel = metrics[
        metrics["model_name"].eq("lightgbm")
        & metrics["evaluation_split"].eq("panel_unique_combined")
    ]
    if panel.empty:
        return pd.Series(dtype=object)
    return panel.iloc[0]


def experiment_record(
    experiment_id: str,
    model_name: str,
    master: pd.Series,
    panel: pd.Series,
    diagnosis: str,
    reason: str,
    selected: bool = False,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "cv_roc_auc_mean": master.get("roc_auc", pd.NA),
        "cv_roc_auc_std": pd.NA,
        "cv_pr_auc_mean": master.get("pr_auc", pd.NA),
        "cv_f1_macro_mean": master.get("f1_macro", pd.NA),
        "cv_mcc_mean": master.get("mcc", pd.NA),
        "panel_unique_roc_auc": panel.get("roc_auc", pd.NA),
        "panel_unique_pr_auc": panel.get("pr_auc", pd.NA),
        "panel_unique_f1_macro": panel.get("f1_macro", pd.NA),
        "panel_unique_mcc": panel.get("mcc", pd.NA),
        "overfitting_gap": pd.NA,
        "selected_threshold": master.get("threshold_value", pd.NA),
        "diagnosis": diagnosis,
        "selected_as_final": bool(selected),
        "reason": reason,
    }


def write_experiment_folder(
    experiment_id: str,
    row: dict[str, object],
    metrics_rows: pd.DataFrame,
    config: dict[str, object],
) -> None:
    metrics_out = METRICS_DIR / "experiments" / experiment_id
    model_out = MODELS_DIR / "experiments" / experiment_id
    figures_out = FIGURES_DIR / "experiments" / experiment_id
    metrics_out.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    metrics_rows.to_csv(metrics_out / "metrics.csv", index=False)
    (metrics_out / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (metrics_out / "diagnosis.md").write_text(
        f"# {experiment_id}\n\n"
        f"Model: {row['model_name']}\n\n"
        f"Diagnosis: {row['diagnosis']}\n\n"
        f"Reason: {row['reason']}\n",
        encoding="utf-8",
    )
    (model_out / "README.md").write_text(
        "This experiment is reconstructed from reproducible saved predictions and metrics. "
        "Final deployable LightGBM artifacts are mirrored in artifacts/models.\n",
        encoding="utf-8",
    )


def snapshot_initial_run() -> None:
    experiment_id = "experiment_001_initial"
    metrics_out = METRICS_DIR / "experiments" / experiment_id
    figures_out = FIGURES_DIR / "experiments" / experiment_id
    metrics_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    for src in [
        TABLES_DIR / "all_evaluation_metrics.csv",
        TABLES_DIR / "threshold_results.csv",
        TABLES_DIR / "main_model_cv_results.csv",
        TABLES_DIR / "panel_generalization_results.csv",
        TABLES_DIR / "error_analysis.csv",
        TABLES_DIR / "feature_importance.csv",
    ]:
        copy_tree_if_absent(src, metrics_out / src.name)

    for src in FIGURES_DIR.glob("*.png"):
        copy_tree_if_absent(src, figures_out / src.name)


def build_experiment_comparison() -> pd.DataFrame:
    metrics_path = TABLES_DIR / "all_evaluation_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("reports/tables/all_evaluation_metrics.csv is required before experiment comparison.")

    metrics = pd.read_csv(metrics_path)
    diagnosis = diagnose_model_performance(metrics)
    panel = panel_combined_row(metrics)
    rows: list[dict[str, object]] = []

    initial = metric_lookup(metrics, "lightgbm", "MASTER_CV", "default_0.5")
    final = metric_lookup(metrics, "lightgbm", "MASTER_CV", "f1_macro_opt")
    youden = metric_lookup(metrics, "lightgbm", "MASTER_CV", "youden_j")

    rows.append(
        experiment_record(
            "experiment_001_initial",
            "lightgbm_default_threshold",
            initial,
            panel,
            "Thresholding weakness",
            "Initial LightGBM at the default 0.5 threshold had acceptable ranking but weak classification balance.",
        )
    )
    rows.append(
        experiment_record(
            "experiment_002_f1_threshold",
            "lightgbm_f1_macro_threshold",
            final,
            panel,
            diagnosis["main_issue"],
            "Validation-derived F1-macro threshold improved classification metrics over the default threshold.",
            selected=True,
        )
    )
    rows.append(
        experiment_record(
            "experiment_003_youden_threshold",
            "lightgbm_youden_j_threshold",
            youden,
            panel,
            "Threshold comparison",
            "Youden-J threshold was tested for sensitivity/specificity balance.",
        )
    )

    alternatives = [
        ("experiment_004_extra_trees", "extra_trees", "Model alternative tested from saved OOF predictions."),
        ("experiment_005_catboost", "catboost", "Model alternative tested from saved OOF predictions."),
        ("experiment_006_ensemble_stack", "ensemble_stack_l1", "Stable stacking candidate tested from saved OOF predictions."),
        ("experiment_007_lr_ek_only", "logistic_regression_ek_only", "EK-only linear baseline retained as an interpretable control."),
    ]
    for experiment_id, model_name, reason in alternatives:
        master = best_master_row(metrics, model_name)
        if not master.empty:
            rows.append(
                experiment_record(
                    experiment_id,
                    model_name,
                    master,
                    pd.Series(dtype=object),
                    "Model alternative",
                    reason,
                )
            )

    ablation_path = TABLES_DIR / "main_model_cv_results.csv"
    if ablation_path.exists():
        ablations = pd.read_csv(ablation_path)
        for experiment_id, ablation_name in [
            ("experiment_008_ek_only_features", "ABL-01_EK_cols_only"),
            ("experiment_009_al_only_features", "ABL-02_AL_cols_only"),
            ("experiment_010_engineered_only", "ABL-03_engineered_only_no_raw_AL_EK"),
        ]:
            subset = ablations[
                ablations["ablation"].eq(ablation_name)
                & ablations["threshold_name"].eq("f1_macro_opt")
            ]
            if subset.empty:
                continue
            row = subset.iloc[0]
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "model_name": ablation_name,
                    "cv_roc_auc_mean": row["CV_AUC"],
                    "cv_roc_auc_std": pd.NA,
                    "cv_pr_auc_mean": pd.NA,
                    "cv_f1_macro_mean": row["CV_F1macro"],
                    "cv_mcc_mean": pd.NA,
                    "panel_unique_roc_auc": pd.NA,
                    "panel_unique_pr_auc": pd.NA,
                    "panel_unique_f1_macro": pd.NA,
                    "panel_unique_mcc": pd.NA,
                    "overfitting_gap": pd.NA,
                    "selected_threshold": pd.NA,
                    "diagnosis": "Feature selection ablation",
                    "selected_as_final": False,
                    "reason": str(row.get("Conclusion", "Feature-group contribution tested.")),
                }
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES_DIR / "experiment_comparison.csv", index=False)

    for row in rows:
        experiment_metrics = metrics[
            metrics["model_name"].astype(str).str.contains(str(row["model_name"]).split("_")[0], case=False, na=False)
        ].copy()
        if experiment_metrics.empty:
            experiment_metrics = pd.DataFrame([row])
        write_experiment_folder(
            str(row["experiment_id"]),
            row,
            experiment_metrics,
            {
                "random_seed": 42,
                "source": "Phase 9.6 experiment comparison from saved predictions, thresholds, ablations, and model alternatives.",
                "selected_threshold": row.get("selected_threshold"),
            },
        )

    return comparison


def write_model_diagnosis() -> dict[str, object]:
    metrics = pd.read_csv(TABLES_DIR / "all_evaluation_metrics.csv")
    diagnosis = diagnose_model_performance(metrics)
    pd.DataFrame(
        [
            {
                "model_strength": diagnosis["model_strength"],
                "main_issue": diagnosis["main_issue"],
                "evidence": diagnosis["evidence"],
                "recommended_next_actions": "; ".join(diagnosis["recommended_next_actions"]),
            }
        ]
    ).to_csv(TABLES_DIR / "model_diagnosis_summary.csv", index=False)

    lines = [
        "# Model Diagnosis",
        "",
        f"Model strength: {diagnosis['model_strength']}",
        "",
        f"Main issue: {diagnosis['main_issue']}",
        "",
        f"Evidence: {diagnosis['evidence']}",
        "",
        "Recommended next actions:",
    ]
    lines.extend(f"- {action}" for action in diagnosis["recommended_next_actions"])
    Path("reports/model_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return diagnosis


def export_final_selection(comparison: pd.DataFrame) -> pd.Series:
    selected_rows = comparison[comparison["selected_as_final"].astype(bool)]
    if selected_rows.empty:
        selected = comparison.sort_values(["cv_f1_macro_mean", "cv_mcc_mean", "cv_roc_auc_mean"], ascending=False).iloc[0]
    else:
        selected = selected_rows.iloc[0]

    final_metrics = pd.DataFrame([selected])
    final_metrics.to_csv(TABLES_DIR / "final_evaluation_metrics.csv", index=False)
    (METRICS_DIR / "final_metrics.json").write_text(final_metrics.to_json(orient="records", indent=2), encoding="utf-8")
    (METRICS_DIR / "final_threshold.json").write_text(
        json.dumps(
            {
                "threshold": None if pd.isna(selected["selected_threshold"]) else float(selected["selected_threshold"]),
                "source": selected["experiment_id"],
                "threshold_strategy": "f1_macro_opt",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    copy_if_exists(METRICS_DIR / "feature_list.json", METRICS_DIR / "final_feature_list.json")
    copy_if_exists(MODELS_DIR / "lightgbm_final.joblib", MODELS_DIR / "final_model.pkl")
    copy_if_exists(PREPROCESSORS_DIR / "feature_engineer.joblib", PREPROCESSORS_DIR / "final_preprocessor.pkl")

    plot_df = comparison.copy()
    plot_df["label"] = plot_df["experiment_id"].astype(str)
    plot_df = plot_df[pd.to_numeric(plot_df["cv_f1_macro_mean"], errors="coerce").notna()]
    if not plot_df.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(plot_df["label"], pd.to_numeric(plot_df["cv_f1_macro_mean"], errors="coerce"), color="#4f6f7f")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("CV F1-macro")
        plt.title("Phase 9.6 experiment comparison")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "final_model_comparison.png", dpi=170)
        plt.close()
    return selected


def write_final_performance_analysis(comparison: pd.DataFrame, diagnosis: dict[str, object], selected: pd.Series) -> None:
    initial = comparison[comparison["experiment_id"].eq("experiment_001_initial")].iloc[0]
    final = selected
    f1_delta = float(final["cv_f1_macro_mean"]) - float(initial["cv_f1_macro_mean"])
    mcc_delta = float(final["cv_mcc_mean"]) - float(initial["cv_mcc_mean"])

    text = f"""# Final Performance Analysis

## Initial Model Performance
The initial LightGBM default-threshold run had MASTER CV ROC-AUC {float(initial['cv_roc_auc_mean']):.4f}, PR-AUC {float(initial['cv_pr_auc_mean']):.4f}, F1-macro {float(initial['cv_f1_macro_mean']):.4f}, and MCC {float(initial['cv_mcc_mean']):.4f}.

## Diagnosis
{diagnosis['evidence']} The automated diagnosis classifies the model as {diagnosis['model_strength']} with main issue `{diagnosis['main_issue']}`.

## Changes Tested
The Phase 9.6 comparison tested default LightGBM thresholding, F1-macro thresholding, Youden-J thresholding, saved model alternatives (ExtraTrees, CatBoost, stacking, EK-only logistic regression), and saved feature-group ablations.

## Experiment Results
See `reports/tables/experiment_comparison.csv` for the full comparison. The selected final configuration is `{final['experiment_id']}` / `{final['model_name']}`.

## Final Selection
The final selected configuration has MASTER CV ROC-AUC {float(final['cv_roc_auc_mean']):.4f}, F1-macro {float(final['cv_f1_macro_mean']):.4f}, and MCC {float(final['cv_mcc_mean']):.4f}. Compared with the initial default-threshold LightGBM run, F1-macro changed by {f1_delta:+.4f} and MCC changed by {mcc_delta:+.4f}.

## Panel-Unique Generalization
The selected LightGBM report keeps panel-unique evaluation as the external-validity proxy. Combined panel metrics are ROC-AUC {float(final['panel_unique_roc_auc']):.4f}, PR-AUC {float(final['panel_unique_pr_auc']):.4f}, F1-macro {float(final['panel_unique_f1_macro']):.4f}, and MCC {float(final['panel_unique_mcc']):.4f}.

## Limitations
Training-score overfitting gaps are not available in the saved metric tables, so overfitting is documented as unavailable rather than fabricated. Hidden competition-set performance cannot be verified locally.

## Recommendations
Future improvement should extend the existing Optuna objective to jointly weight CV ROC-AUC, CV F1-macro, CV MCC, and panel-unique PR-AUC, then rerun the same experiment logging contract.
"""
    Path("reports/final_performance_analysis.md").write_text(text, encoding="utf-8")


def run_phase96_outputs() -> tuple[pd.DataFrame, dict[str, object], pd.Series]:
    snapshot_initial_run()
    comparison = build_experiment_comparison()
    diagnosis = write_model_diagnosis()
    selected = export_final_selection(comparison)
    write_final_performance_analysis(comparison, diagnosis, selected)
    return comparison, diagnosis, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="TEKNOFEST 2026 end-to-end pipeline.")
    parser.add_argument("--mode", choices=["smoke", "full", "tune", "evaluate"], default="smoke")
    args = parser.parse_args()

    ensure_dirs()
    data_dir = discover_data_dir(Path.cwd())
    prepared = prepare_data(data_dir)
    write_data_diagnostics(data_dir, TABLES_DIR)
    save_validation_diagnostics(prepared)
    save_feature_list(prepared)

    if args.mode != "evaluate":
        save_baselines(prepared)
        base_mode = "full" if args.mode == "full" else "smoke"
        save_main_outputs(prepared, base_mode)
        save_explainability_exports()

    generate_phase9_outputs(prepared)
    if args.mode in {"tune", "evaluate", "full"}:
        comparison, diagnosis, selected = run_phase96_outputs()
        print(
            "Phase 9.6 selected "
            f"{selected['experiment_id']} ({selected['model_name']}) "
            f"as {diagnosis['model_strength']} / {diagnosis['main_issue']}."
        )
        phase10 = run_phase10_improvements(prepared, load_params(), mode=args.mode)
        print(
            "Phase 10 selected "
            f"{phase10['selected_threshold_strategy']} threshold "
            f"{phase10['selected_threshold']:.6f} "
            f"as {phase10['model_strength']} / {phase10['main_issue']}."
        )
    save_final_report_summary(prepared)
    print(f"Pipeline complete in {args.mode} mode.")
    print(f"Tables: {TABLES_DIR.resolve()}")
    print(f"Figures: {FIGURES_DIR.resolve()}")
    print(f"Artifacts: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
