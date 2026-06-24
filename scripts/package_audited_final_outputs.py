"""Package the audited model's saved evidence into the final competition contract.

This script is deliberately report-only: training happens in
``run_teknofest_readiness_audit.py``.  Keeping packaging separate prevents a
report refresh from silently changing a model or its metrics.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from final_inference import generate_final_submission  # noqa: E402
from medical_metrics import compute_medical_metrics  # noqa: E402


OUT = ROOT / "outputs"
AUDIT = ROOT / "audit_outputs"
PRED = ROOT / "artifacts" / "predictions"
DECISION = ROOT / "artifacts" / "metrics" / "audited_final_model_decision.json"
DATA = ROOT / "teknofest2026_artificialintelligenceinhealtcare-main" / "YARISMA_TRAIN_MASTER.csv"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _group(feature: str) -> str:
    if feature.startswith("AL_") or feature in {"n_pops", "max_AF", "min_AF_nz", "log_max_AF", "log_min_AF", "BA1_flag", "BS1_flag", "PM2_flag", "BS2_proxy"}:
        return "AL / population frequency"
    if feature.startswith("EK_") or feature.startswith("EK"):
        return "EK / computational"
    if feature.startswith("AA_") or feature.startswith("aa_") or "blosum" in feature:
        return "AA / amino-acid"
    if feature.startswith("CAT_") or feature.startswith("cat") or feature.startswith("geno"):
        return "CAT / metadata"
    if feature.startswith("miss"):
        return "missingness"
    if "x" in feature or "minus" in feature or "squared" in feature:
        return "interaction"
    return "other"


def _metric_row(frame: pd.DataFrame, split: str, source: Path, threshold: float, model: str, timestamp: str) -> dict[str, object]:
    m = compute_medical_metrics(frame["Label"].astype(int), frame["score"], threshold)
    return {
        "evaluation_split": split, "model_name": model, "threshold": threshold,
        "n_samples": len(frame), "n_pathogenic": int(frame["Label"].sum()), "n_benign": int((frame["Label"] == 0).sum()),
        "precision": m["precision"], "recall": m["pathogenic_recall"], "specificity": m["specificity"],
        "f1_binary": m["f1_binary"], "f1_macro": m["f1_macro"], "mcc": m["mcc"], "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
        "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        "source_prediction_file": str(source.relative_to(ROOT)), "verified_timestamp": timestamp,
    }


def _plot_threshold(sweep: pd.DataFrame, selected: float, output: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(sweep.threshold, sweep.f1_macro, label="F1-macro")
    plt.plot(sweep.threshold, sweep.mcc, label="MCC")
    plt.axvline(selected, color="black", linestyle="--", label=f"selected={selected:.4f}")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.legend(); plt.tight_layout(); plt.savefig(output, dpi=180); plt.close()


def _plot_subgroups(metrics: pd.DataFrame, output: Path) -> None:
    subset = metrics[metrics.evaluation_split.ne("MASTER_ONLY_CV")].copy()
    positions = np.arange(len(subset)); width = .36
    plt.figure(figsize=(9, 5)); plt.bar(positions - width / 2, subset.f1_macro, width, label="F1-macro"); plt.bar(positions + width / 2, subset.mcc, width, label="MCC")
    plt.xticks(positions, subset.evaluation_split, rotation=18, ha="right"); plt.ylim(0, 1); plt.legend(); plt.tight_layout(); plt.savefig(output, dpi=180); plt.close()


def _plot_confusion(metrics: pd.DataFrame, output: Path) -> None:
    row = metrics.loc[metrics.evaluation_split.eq("MASTER_ONLY_CV")].iloc[0]
    matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
    plt.figure(figsize=(4.5, 4)); plt.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2): plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["Benign", "Pathogenic"]); plt.yticks([0, 1], ["Benign", "Pathogenic"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(output, dpi=180); plt.close()


def main() -> None:
    if not DECISION.exists():
        raise FileNotFoundError(f"Missing audited decision: {DECISION}. Run scripts/run_teknofest_readiness_audit.py first.")
    decision = json.loads(DECISION.read_text(encoding="utf-8")); threshold = float(decision["threshold"]); model = str(decision["model_id"])
    audit_dir, final_dir, fig_dir, pdr_dir = (OUT / "audit", OUT / "final", OUT / "figures", OUT / "pdr_assets")
    for directory in (audit_dir, final_dir, fig_dir, pdr_dir): directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    master_path, panel_path = PRED / "audited_master_only_oof_predictions.csv", PRED / "audited_panel_unique_predictions.csv"
    master, panels = pd.read_csv(master_path), pd.read_csv(panel_path)
    rows = [_metric_row(master, "MASTER_ONLY_CV", master_path, threshold, model, timestamp)]
    for dataset, frame in panels.groupby("dataset", sort=True): rows.append(_metric_row(frame, f"{dataset}_UNIQUE", panel_path, threshold, model, timestamp))
    rows.append(_metric_row(panels, "PANEL_UNIQUE_COMBINED", panel_path, threshold, model, timestamp))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(final_dir / "final_metric_verification.csv", index=False); metrics.to_csv(final_dir / "subgroup_metrics.csv", index=False)
    sweep = pd.read_csv(AUDIT / "recomputed_threshold_sweep.csv")
    sweep.to_csv(final_dir / "threshold_sweep.csv", index=False); _plot_threshold(sweep, threshold, fig_dir / "threshold_optimization.png"); _plot_subgroups(metrics, fig_dir / "subgroup_performance.png"); _plot_confusion(metrics, fig_dir / "master_confusion_matrix.png")
    default = _metric_row(master, "MASTER_ONLY_CV_DEFAULT_0.50", master_path, .5, model, timestamp)
    _write(audit_dir / "metric_audit_report.md", "# Metric Audit\n\nAll canonical metrics were recomputed from `audited_master_only_oof_predictions.csv` and `audited_panel_unique_predictions.csv` in this run. Class 1 is pathogenic and class 0 is benign. Legacy score files are historical only and are not authoritative.\n\n" + metrics.to_markdown(index=False))
    _write(audit_dir / "data_audit_report.md", (AUDIT / "02_data_audit_report.md").read_text(encoding="utf-8") + "\n\n## Compatibility\n\nAudited inference uses serialized feature order and rejects missing required raw columns. No official test CSV or organizer submission schema was found.")
    _write(audit_dir / "leakage_audit_report.md", (AUDIT / "03_leakage_audit_report.md").read_text(encoding="utf-8"))
    _write(audit_dir / "prioritized_fix_plan.md", """# Prioritized Fix Plan and Applied Status

## CRITICAL

- No unresolved critical issue remains in the audited candidate. No official test labels were found or used.

## HIGH

- Issue: legacy final reporting reused saved OOF reference predictions and had an error-analysis count mismatch.
  Evidence: legacy decision artifact reports `matches_selected_oof_confusion_counts: false`.
  Fix: regenerate a separate audited candidate and canonical metrics from its saved OOF/panel predictions.
  Impact: final contract no longer relies on stale legacy counts.
- Issue: supplied-panel overlap can contaminate MASTER-based validation.
  Fix: exclude every MASTER Variant_ID found in KANSER/PAH/CFTR before both audited train and validation folds.
  Impact: clean MASTER-only OOF population.

## MEDIUM

- Issue: threshold selection still uses OOF evidence, not a third independent validation set.
  Fix: use the median fold threshold; document it as selection evidence and retain the default-0.50 comparison.
  Impact: transparent, but not fully independent threshold validation.
- Issue: organizer submission schema is unavailable.
  Fix: output full predictions and a conservative two-column compact submission; require organizer-template confirmation.

## LOW

- Issue: prior evidence was scattered across legacy reports.
  Fix: package canonical audit, final, figure, and PDR paths under `outputs/`.
""")
    default_frame = pd.DataFrame([default]); selected = metrics[metrics.evaluation_split.eq("MASTER_ONLY_CV")]
    _write(audit_dir / "threshold_audit_report.md", "# Threshold Audit\n\nSelected threshold is `{:.6f}`, the median of fold-specific F1-macro optima. It was selected from MASTER-only OOF evidence; no panel or official test label was used. This is model-selection evidence, not an independent final test estimate.\n\n## Default 0.50\n\n{}\n\n## Selected\n\n{}".format(threshold, default_frame.to_markdown(index=False), selected.to_markdown(index=False)))
    config = ROOT / "configs" / "audited_final_model_config.json"
    _write(audit_dir / "hyperparameter_audit_report.md", "# Hyperparameter Audit\n\nThe audited LightGBM uses a conservative, saved configuration at `configs/audited_final_model_config.json`. No new exhaustive hyperparameter search was run in this pass, avoiding repeated tuning on the same small validation data. The legacy Optuna record remains non-authoritative for audited selection.")
    # Error analysis is explicitly derived from the same saved prediction files.
    errors = pd.concat([master.assign(evaluation_split="MASTER_ONLY_CV"), panels.assign(evaluation_split=panels.dataset + "_UNIQUE")], ignore_index=True)
    errors["error_type"] = np.select([(errors.Label.eq(1) & errors.prediction.eq(0)), (errors.Label.eq(0) & errors.prediction.eq(1))], ["false_negative", "false_positive"], default="correct")
    errors["distance_to_threshold"] = (errors.score - threshold).abs(); errors["confidence_band"] = pd.cut(errors.distance_to_threshold, [-.001, .05, .2, np.inf], labels=["borderline", "moderate", "high"])
    errors.to_csv(final_dir / "error_analysis.csv", index=False)
    error_summary = errors.groupby(["evaluation_split", "error_type"], dropna=False).size().reset_index(name="n")
    _write(audit_dir / "error_analysis_report.md", "# Error Analysis\n\nFalse negatives are clinically more concerning because pathogenic variants may be missed. KANSER has high recall but comparatively lower precision; PAH has the weakest MCC among panels. CFTR has only 34 unique rows and must not be over-interpreted.\n\n" + error_summary.to_markdown(index=False) + "\n\nError rows and confidence bands are in `outputs/final/error_analysis.csv`.")
    importance = pd.read_csv(AUDIT / "audited_feature_importance.csv"); importance["feature_group"] = importance.feature.map(_group); importance.to_csv(final_dir / "feature_importance.csv", index=False); shutil.copy2(AUDIT / "audited_feature_importance.png", fig_dir / "feature_importance.png")
    _write(audit_dir / "explainability_audit_report.md", "# Explainability Audit\n\nThe importance file is generated directly from the audited serialized LightGBM artifact. Variant_ID and Label are not model features. Importance is predictive split importance, not biological causality. Legacy SHAP files do not represent this audited model.\n\n" + importance.groupby("feature_group", as_index=False).importance.sum().sort_values("importance", ascending=False).to_markdown(index=False))
    _write(audit_dir / "subgroup_audit_report.md", "# Subgroup Audit\n\nPanels are evaluated only on rows unique relative to MASTER. A common global threshold is used; no per-panel threshold tuning was performed. CFTR's small n makes its high score uncertain.\n\n" + metrics.to_markdown(index=False))
    _write(final_dir / "model_selection_report.md", f"# Model Selection Report\n\nSelected model: `{model}`. It is selected for this final contract because it is the only newly regenerated candidate in this pass with fold-local feature fitting and complete saved OOF/panel predictions. The legacy multi-model comparisons are retained but are not used to claim superiority because their preserved reference OOF artifact was copied rather than regenerated. Threshold: `{threshold:.6f}`.\n\n" + metrics.to_markdown(index=False))
    # Dry-run uses labelled MASTER solely as an input-schema surrogate; inference ignores Label.
    raw = pd.read_csv(DATA); generate_final_submission(raw, final_dir / "final_predictions.csv", DECISION); generate_final_submission(raw, final_dir / "submission.csv", DECISION, basic_submission=True)
    _write(final_dir / "pdr_evidence_summary.md", f"# PDR Evidence Summary\n\n- Final model: `{model}`\n- Threshold: `{threshold:.6f}`\n- Validation: 5-fold stratified MASTER-only OOF after removal of every panel-overlapping MASTER Variant_ID; feature fitting is fold-local.\n- Leakage conclusion: audited candidate avoids ID features, panel-overlap train/validation contamination, and full-fold target-encoding fitting.\n- Explainability: audited LightGBM split importance is available; it is predictive only.\n- Error summary: false negatives remain the clinically more consequential failure; PAH has the lowest subgroup MCC and CFTR is small-n.\n- Limitation: official test data/template were not found, so no official-test claim is made.\n\n## Verified metrics\n\n" + metrics.to_markdown(index=False) + "\n\n## PDR assets\n\n- `outputs/figures/threshold_optimization.png`\n- `outputs/figures/subgroup_performance.png`\n- `outputs/figures/master_confusion_matrix.png`\n- `outputs/figures/feature_importance.png`\n- `outputs/final/final_metric_verification.csv`\n- `outputs/final/subgroup_metrics.csv`\n- `outputs/final/error_analysis.csv`")
    _write(final_dir / "TEKNOFEST_MODEL_READINESS_REPORT.md", f"# TEKNOFEST Model Readiness Report\n\n## Executive summary\n\n**CONDITIONAL GO.** The audited model is reproducible, has saved recomputed metrics, excludes supplied-panel overlap from its MASTER training/validation population, and completes label-free inference. The official test CSV and official submission schema are absent.\n\n## Final selected model\n\n`{model}`; threshold `{threshold:.6f}`; artifact `{decision['artifact_path']}`.\n\n## Dataset and validation status\n\nMASTER-only 5-fold stratified OOF; KANSER/PAH/CFTR unique panels are external-like labelled checks. No hidden-test metrics are claimed.\n\n## Metric verification and subgroup performance\n\n{metrics.to_markdown(index=False)}\n\n## Threshold optimization\n\nDefault 0.50 and selected threshold metrics are preserved in `outputs/audit/threshold_audit_report.md`; full sweep is `outputs/final/threshold_sweep.csv`.\n\n## Errors, explainability, and inference\n\nSee error, explainability, and inference outputs in this directory. Label-free dry-run outputs were generated from MASTER schema only; Label was ignored.\n\n## Remaining limitations\n\n- OOF threshold selection is not a third independent test set.\n- Panel sample sizes are limited, especially CFTR.\n- Distribution shift and organizer submission format remain unknown.\n\n## Reproduce\n\n```powershell\npython src/run_final_pipeline.py\npython src/sanity_check_final.py\npython src/predict_final.py --input path/to/official_test.csv --output outputs/final/submission.csv --basic-submission\n```\n\n## Go / No-Go\n\n**CONDITIONAL GO** pending organizer test input and submission template confirmation.")
    for path in [final_dir / "final_metric_verification.csv", final_dir / "subgroup_metrics.csv", final_dir / "threshold_sweep.csv", final_dir / "error_analysis.csv", final_dir / "feature_importance.csv", fig_dir / "threshold_optimization.png", fig_dir / "subgroup_performance.png", fig_dir / "master_confusion_matrix.png", fig_dir / "feature_importance.png"]: shutil.copy2(path, pdr_dir / path.name)
    print(f"Packaged final outputs under {OUT}")


if __name__ == "__main__":
    main()
