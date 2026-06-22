from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import TABLES_DIR
from data_loading import discover_data_dir
from phase10_improvements import run_phase10_improvements
from phase9_outputs import generate_phase9_outputs
from run_pipeline import load_params
from teknofest.data_prep import prepare_data


def _fmt(value: object) -> str:
    if pd.isna(value):
        return "not available"
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return str(value)


def write_performance_improvement_report() -> Path:
    selection = pd.read_csv(TABLES_DIR / "final_model_selection_table.csv")
    final = selection[selection["selected_as_final"].astype(bool)].iloc[0]
    before_after = pd.read_csv(TABLES_DIR / "phase10_before_after_comparison.csv")
    previous = before_after[before_after["model_profile"].eq("previous_threshold_optimized")].iloc[0]
    current = before_after[before_after["selected_as_final"].astype(bool)].iloc[0]
    calibration = pd.read_csv(TABLES_DIR / "calibration_decision_matrix.csv")
    thresholds = pd.read_csv(TABLES_DIR / "advanced_threshold_comparison.csv")
    metrics = pd.read_csv(TABLES_DIR / "all_evaluation_metrics.csv")
    feature_ablation = pd.read_csv(TABLES_DIR / "feature_group_ablation_results.csv")
    error_summary = pd.read_csv(TABLES_DIR / "final_error_group_feature_summary.csv")
    panel = pd.read_csv(TABLES_DIR / "final_panel_specific_metrics.csv")

    master_lightgbm = metrics[
        metrics["model_name"].eq("lightgbm")
        & metrics["evaluation_split"].eq("MASTER_CV")
        & metrics["threshold_type"].eq("f1_macro_opt")
    ].iloc[0]
    best_cal = calibration.sort_values("brier_score").iloc[0]
    selected_thresholds = thresholds[
        thresholds["model_name"].eq("lightgbm")
        & thresholds["evaluation_split"].eq("MASTER_CV")
        & thresholds["threshold_strategy"].isin(["default_0.5", "f1_macro_opt", "mcc_opt", "youden_j", "balanced_accuracy_opt"])
    ][["threshold_strategy", "threshold", "f1_macro", "mcc", "recall", "specificity"]]
    computed_ablation = feature_ablation[feature_ablation["status"].eq("computed_existing_ablation")]
    fn_count = int(pd.read_csv(TABLES_DIR / "final_false_negative_cases.csv").shape[0])
    fp_count = int(pd.read_csv(TABLES_DIR / "final_false_positive_cases.csv").shape[0])

    text = f"""# Model Performance Improvement Report

## 1. Current Baseline Performance
The current LightGBM OOF baseline at the F1-macro threshold has ROC-AUC {_fmt(master_lightgbm['roc_auc'])}, PR-AUC {_fmt(master_lightgbm['pr_auc'])}, F1-macro {_fmt(master_lightgbm['f1_macro'])}, MCC {_fmt(master_lightgbm['mcc'])}, pathogenic recall {_fmt(master_lightgbm['recall'])}, and specificity {_fmt(master_lightgbm['specificity'])}.

## 2. Improved Model Performance
The selected final profile is `{final['profile']}` with threshold strategy `{final['threshold_strategy']}` and threshold {_fmt(final['threshold'])}. MASTER metrics are ROC-AUC {_fmt(final['roc_auc'])}, PR-AUC {_fmt(final['pr_auc'])}, F1-macro {_fmt(final['f1_macro'])}, and MCC {_fmt(final['mcc'])}. Panel-combined metrics are ROC-AUC {_fmt(final['panel_roc_auc'])}, PR-AUC {_fmt(final['panel_pr_auc'])}, F1-macro {_fmt(final['panel_f1_macro'])}, and MCC {_fmt(final['panel_mcc'])}.

## 3. Before vs After Comparison
Previous threshold-optimized profile: MASTER F1-macro {_fmt(previous['master_f1_macro'])}, MCC {_fmt(previous['master_mcc'])}, panel F1-macro {_fmt(previous['panel_f1_macro'])}, panel MCC {_fmt(previous['panel_mcc'])}.

Final selected profile: MASTER F1-macro {_fmt(current['master_f1_macro'])}, MCC {_fmt(current['master_mcc'])}, panel F1-macro {_fmt(current['panel_f1_macro'])}, panel MCC {_fmt(current['panel_mcc'])}.

## 4. Best Single Model
Best single-model recommendation: LightGBM with the `{final['profile']}` configuration. It was selected because it improves the final selection score after considering decision metrics, panel behavior, overfitting gap, and threshold stability.

## 5. Best Ensemble Model
The saved OOF ensemble stack was evaluated and is included in `reports/tables/all_evaluation_metrics.csv` and `reports/tables/experiment_comparison.csv`. It improves OOF ranking/decision metrics over several baselines, but the final recommendation remains LightGBM because the final selected LightGBM profile has explicit panel evaluation and final artifact support.

## 6. Best Threshold
Recommended final threshold: {_fmt(final['threshold'])}. Threshold comparison:

{selected_thresholds.to_markdown(index=False)}

## 7. Best Calibration Method
Best MASTER Brier score method: `{best_cal['calibration_method']}` with Brier {_fmt(best_cal['brier_score'])}. Calibration is reported but not selected for the final decision model because panel trade-offs and decision metrics did not justify replacing the uncalibrated decision scores. See `reports/calibration_decision_review.md`.

## 8. Feature Selection Decision
Feature ablations were evaluated where saved OOF results exist. Computed feature ablations:

{computed_ablation[['configuration', 'master_cv_roc_auc', 'master_cv_f1_macro', 'generalization_flag']].to_markdown(index=False)}

Top-k and some subgroup-only ablations remain explicitly marked as queued/not retrained; their metrics were not fabricated.

## 9. Leakage Audit Summary
Validation remains contamination-aware: MASTER variants shared with panels are excluded from validation folds and panel-unique subsets are evaluated separately. Variant_ID is not used as a predictive feature. Leakage and feature-quality outputs are available in EDA and Phase 9/10 reports; no perfect-separation leakage feature was accepted as final.

## 10. Medical Metric Report
The complete medical metric table is `reports/tables/all_evaluation_metrics.csv`. It includes ROC-AUC, PR-AUC, accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, F1-macro, F1-weighted, MCC, log loss, Brier score, and confusion matrix counts.

## 11. Error Analysis Summary
Final error analysis found {fn_count} false-negative rows and {fp_count} false-positive rows across saved final predictions. Feature summaries by TP/TN/FP/FN are saved in `reports/tables/final_error_group_feature_summary.csv`. Panel-specific final metrics:

{panel[['evaluation_split', 'roc_auc', 'pr_auc', 'f1_macro', 'mcc', 'recall', 'specificity', 'panel_role']].to_markdown(index=False)}

## 12. Final Hidden-Test Strategy
Use the selected LightGBM final artifacts under `artifacts/models` and `artifacts/preprocessors`, apply the saved final threshold in `artifacts/metrics/final_threshold.json`, preserve calibration as report-only unless the competition format specifically requires calibrated probabilities, and prioritize panel-unique robustness over small OOF-only gains.

## Rejected Or Deferred Changes
Calibration was not selected for final decision scoring because it improved some probability losses but did not clearly preserve panel decision behavior. Uncomputed feature-group experiments are marked as queued rather than reported. Accuracy-only selection was rejected in favor of medical metrics.
"""
    out = PROJECT_ROOT / "reports" / "model_performance_improvement_report.md"
    out.write_text(text, encoding="utf-8")
    return out


def write_model_performance_audit() -> Path:
    metrics = pd.read_csv(TABLES_DIR / "all_evaluation_metrics.csv")
    selection = pd.read_csv(TABLES_DIR / "final_model_selection_table.csv")
    stability = pd.read_csv(TABLES_DIR / "fold_threshold_stability.csv")
    calibration = pd.read_csv(TABLES_DIR / "calibration_comparison.csv")
    leakage = pd.read_csv(PROJECT_ROOT / "reports" / "eda" / "tables" / "phase4_leakage_suspects.csv")

    current = metrics[
        metrics["model_name"].eq("lightgbm")
        & metrics["evaluation_split"].eq("MASTER_CV")
        & metrics["threshold_type"].eq("f1_macro_opt")
    ].iloc[0]
    default = metrics[
        metrics["model_name"].eq("lightgbm")
        & metrics["evaluation_split"].eq("MASTER_CV")
        & metrics["threshold_type"].eq("default_0.5")
    ].iloc[0]
    final = selection[selection["selected_as_final"].astype(bool)].iloc[0]
    best_cal = calibration[calibration["evaluation_split"].eq("MASTER_CV")].sort_values("brier_score").iloc[0]
    f1_stability = stability[stability["threshold_strategy"].eq("f1_macro_opt")]
    threshold_std = f1_stability["std"].dropna().iloc[0] if not f1_stability.empty else np.nan

    text = f"""# Model Performance Audit

## Current Best Model
Current saved baseline model: LightGBM at F1-macro optimized threshold. Selected improved model: LightGBM `{final['profile']}` at threshold {_fmt(final['threshold'])}.

## Current Validation Strategy
The pipeline uses contamination-aware cross-validation on MASTER, excluding MASTER variants shared with panels from validation folds, plus panel-unique external checks for KANSER, PAH, and CFTR.

## Current Metrics
Default threshold LightGBM: F1-macro {_fmt(default['f1_macro'])}, MCC {_fmt(default['mcc'])}, recall {_fmt(default['recall'])}, specificity {_fmt(default['specificity'])}.

F1-threshold LightGBM: ROC-AUC {_fmt(current['roc_auc'])}, PR-AUC {_fmt(current['pr_auc'])}, F1-macro {_fmt(current['f1_macro'])}, MCC {_fmt(current['mcc'])}, recall {_fmt(current['recall'])}, specificity {_fmt(current['specificity'])}.

## Audit Questions

1. Where is the model currently weak?
   The default threshold is weak: it gives very high pathogenic recall but poor specificity and low MCC. Calibration also needs caution.

2. Which metric is the biggest problem?
   At default threshold, MCC and specificity are the biggest issues. After threshold optimization, remaining weakness is moderate MCC/F1 rather than ranking ability.

3. Is the model underfitting or overfitting?
   The selected profile was chosen partly to reduce overfitting gap. See `reports/tables/overfitting_gap_analysis.csv`; hidden-test performance remains unknown.

4. Are there fold-to-fold performance instabilities?
   Threshold stability is tracked in `reports/tables/fold_threshold_stability.csv`. F1 threshold std is {_fmt(threshold_std)}.

5. Are there class-specific weaknesses?
   Yes. The default threshold over-predicts pathogenic class, hurting benign specificity. Threshold optimization improves benign specificity while keeping pathogenic recall clinically high.

6. Is Pathogenic recall strong enough?
   F1-threshold LightGBM pathogenic recall is {_fmt(current['recall'])}. Final panel-specific recalls are documented in `reports/tables/final_panel_specific_metrics.csv`.

7. Are probabilities calibrated?
   Calibration was tested. Best MASTER Brier method is `{best_cal['calibration_method']}` with Brier {_fmt(best_cal['brier_score'])}; calibration remains report-only because panel trade-offs were not clearly favorable.

8. Are any features suspiciously predictive?
   Leakage scan table has {len(leakage)} rows. No direct label proxy was accepted as final, and Variant_ID is excluded from modeling.

9. Is validation realistic for hidden-test performance?
   It is more realistic than simple random split because it uses contamination-aware MASTER CV and panel-unique evaluation. It is still not a guarantee of hidden-test performance.

## Audit Conclusion
The model is not weak, but it should be conservatively described as moderate-to-good. The main improvement opportunity was thresholding and clinically weighted selection, not raw accuracy maximization.
"""
    out = PROJECT_ROOT / "reports" / "model_performance_audit.md"
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model performance improvement plan.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["evaluate", "full", "tune"], default="evaluate")
    parser.add_argument("--refresh-phase9", action="store_true")
    parser.add_argument("--reports-only", action="store_true")
    parser.add_argument("--competition-final", action="store_true", help="Run the complete final model zoo, ensemble, selection, inference, and report workflow.")
    parser.add_argument("--skip-repeated-validation", action="store_true", help="Skip the repeated-seed validation stage only when iterating locally.")
    args = parser.parse_args()

    if args.reports_only:
        audit = write_model_performance_audit()
        out = write_performance_improvement_report()
        print(f"Audit written to: {audit.resolve()}")
        print(f"Report written to: {out.resolve()}")
        return

    data_dir = args.data_dir or discover_data_dir(PROJECT_ROOT)
    prepared = prepare_data(data_dir)
    if args.competition_final:
        from final_competition_pipeline import run_final_competition_pipeline

        result = run_final_competition_pipeline(prepared, not args.skip_repeated_validation)
        print("Final competition workflow complete.")
        print(f"Selected candidate: {result['selected']['candidate_id']}")
        print(f"Submission: {result['submission_path']}")
        print(f"Report assets: {result['report_assets']}")
        return
    if args.refresh_phase9 or not (TABLES_DIR / "all_evaluation_metrics.csv").exists():
        generate_phase9_outputs(prepared)
    result = run_phase10_improvements(prepared, load_params(), mode=args.mode)
    audit = write_model_performance_audit()
    out = write_performance_improvement_report()
    print("Model performance improvement complete.")
    print(f"Strength: {result['model_strength']}")
    print(f"Main issue: {result['main_issue']}")
    print(f"Selected threshold: {result['selected_threshold']:.6f}")
    print(f"Threshold strategy: {result['selected_threshold_strategy']}")
    print(f"Audit: {audit.resolve()}")
    print(f"Report: {out.resolve()}")


if __name__ == "__main__":
    main()
