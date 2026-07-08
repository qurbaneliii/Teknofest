"""Copy and document the protected final baseline without mutating its sources."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "artifacts" / "final_locked"
REPORTS = ROOT / "reports" / "final_model_lock"
TABLES = REPORTS / "tables"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy_artifact(name: str, source: Path, destination: Path, kind: str, inference: bool, metric: bool, notes: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"artifact_name": name, "original_path": str(source), "locked_copy_path": str(destination), "artifact_type": kind, "sha256_original": digest(source), "sha256_locked_copy": digest(destination), "size_bytes": source.stat().st_size, "created_or_modified_time": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(), "required_for_inference": inference, "required_for_metric_verification": metric, "protected_do_not_modify": True, "notes": notes}


def main() -> None:
    for folder in (LOCK / "model", LOCK / "preprocessing", LOCK / "predictions", LOCK / "metadata", REPORTS, TABLES):
        folder.mkdir(parents=True, exist_ok=True)
    artifacts = [
        ("final_model", ROOT / "artifacts/models/final_model.pkl", LOCK / "model/final_model.pkl", "final model bundle", True, False, "Selected protected LightGBM bundle."),
        ("final_model_columns", ROOT / "artifacts/models/final_model_columns.txt", LOCK / "model/final_model_columns.txt", "feature schema", True, True, "Exact aligned model columns."),
        ("final_preprocessor", ROOT / "artifacts/preprocessors/final_preprocessor.pkl", LOCK / "preprocessing/final_preprocessor.pkl", "preprocessing artifact", True, False, "Protected feature engineer."),
        ("final_threshold", ROOT / "artifacts/metrics/final_threshold.json", LOCK / "metadata/final_threshold.json", "threshold metadata", True, True, "Threshold must remain 0.471."),
        ("final_metrics", ROOT / "artifacts/metrics/final_metrics.json", LOCK / "metadata/final_metrics.json", "reported metrics", False, True, "Source metrics for reverification."),
        ("final_decision_source", ROOT / "artifacts/metrics/final_model_decision.json", LOCK / "metadata/final_model_decision_source.json", "selection decision", True, True, "Protected final selection decision."),
        ("final_feature_list", ROOT / "artifacts/metrics/final_feature_list.json", LOCK / "metadata/final_feature_list.json", "feature schema", True, True, "Feature leakage audit input."),
        ("final_config", ROOT / "artifacts/metrics/final_config.json", LOCK / "metadata/final_config.json", "model configuration", False, True, "Selected model configuration."),
        ("master_oof_predictions", ROOT / "artifacts/predictions/final_master_cv_predictions.csv", LOCK / "predictions/final_master_cv_predictions.csv", "MASTER OOF predictions", False, True, "Metric evidence."),
        ("panel_predictions", ROOT / "artifacts/predictions/final_panel_predictions.csv", LOCK / "predictions/final_panel_predictions.csv", "panel-unique predictions", False, True, "Panel metric evidence."),
        ("selection_table", ROOT / "reports/tables/final_model_selection_table.csv", LOCK / "metadata/final_model_selection_table.csv", "selection table", False, True, "Final selection evidence."),
        ("metric_audit", ROOT / "reports/tables/final_metric_verification_audit.csv", LOCK / "metadata/final_metric_verification_audit.csv", "metric audit", False, True, "Saved audit evidence."),
        ("panel_specific_metrics", ROOT / "reports/tables/final_panel_specific_metrics.csv", LOCK / "metadata/final_panel_specific_metrics.csv", "panel metrics", False, True, "CFTR/KANSER/PAH evidence."),
        ("medical_metric_comparison", ROOT / "reports/tables/final_medical_metric_comparison.csv", LOCK / "metadata/final_medical_metric_comparison.csv", "medical metrics", False, True, "Final clinical-metric comparison."),
        ("metric_audit_report", ROOT / "reports/final_metric_verification_audit.md", LOCK / "metadata/final_metric_verification_audit.md", "evidence report", False, True, "Human-readable audit."),
        ("inference_source", ROOT / "src/final_inference.py", LOCK / "metadata/final_inference_source.py", "inference source", True, False, "Source implementation used by locked wrapper."),
        ("requirements", ROOT / "requirements.txt", LOCK / "metadata/requirements.txt", "environment", True, False, "Pinned environment dependencies."),
        ("locked_inference_script", ROOT / "scripts/generate_final_predictions.py", LOCK / "metadata/generate_final_predictions.py", "inference CLI", True, False, "Locked inference entry point."),
    ]
    rows, missing = [], []
    for name, source, target, kind, inference, metric, notes in artifacts:
        if source.exists():
            rows.append(copy_artifact(name, source, target, kind, inference, metric, notes))
        else:
            missing.append({"artifact_name": name, "expected_path": str(source), "blocking": bool(inference or metric), "reason": "expected protected evidence file was not found"})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(TABLES / "final_artifact_manifest.csv", index=False)
    (LOCK / "metadata/final_artifact_manifest.json").write_text(manifest.to_json(orient="records", indent=2), encoding="utf-8")
    pd.DataFrame(missing, columns=["artifact_name", "expected_path", "blocking", "reason"]).to_csv(TABLES / "missing_final_artifacts.csv", index=False)
    source_decision = json.loads((ROOT / "artifacts/metrics/final_model_decision.json").read_text(encoding="utf-8"))
    locked_decision = {"model_id": "lightgbm_conservative_regularized", "threshold": 0.471, "threshold_strategy": "profile_f1_macro_opt", "calibration": "none", "ensemble_replaced_lightgbm": False, "model_bundle": "model/final_model.pkl", "preprocessor": "preprocessing/final_preprocessor.pkl", "feature_schema": "model/final_model_columns.txt", "master_oof_predictions": "predictions/final_master_cv_predictions.csv", "panel_predictions": "predictions/final_panel_predictions.csv", "source_decision": source_decision}
    (LOCK / "metadata/final_model_decision.json").write_text(json.dumps(locked_decision, indent=2), encoding="utf-8")
    oof, panel = source_decision["oof_metrics"], source_decision["panel_unique_combined_metrics"]
    stress = pd.read_csv(ROOT / "reports/v3/stress_validation/tables/robust_selection_decision.csv").iloc[0]
    decision_md = f"""# Final Model Decision

The protected LightGBM baseline is locked as the final model.

- Model ID: `lightgbm_conservative_regularized`
- The final threshold is 0.471.
- Threshold strategy: `profile_f1_macro_opt`.
- No calibration is used.
- No ensemble replaces the final LightGBM model.

## Verified evidence

MASTER OOF: ROC-AUC {oof['roc_auc']:.4f}; PR-AUC {oof['pr_auc']:.4f}; F1-macro {oof['f1_macro']:.4f}; MCC {oof['mcc']:.4f}.

Panel-unique combined: ROC-AUC {panel['roc_auc']:.4f}; PR-AUC {panel['pr_auc']:.4f}; F1-macro {panel['f1_macro']:.4f}; MCC {panel['mcc']:.4f}.

## Candidate disposition

HistGradientBoosting remains exploratory and rejected for final replacement. Its repeated MASTER F1-macro ({stress['master_f1_macro_mean']:.4f}) and MCC ({stress['master_mcc_mean']:.4f}) did not establish improvement over the protected OOF profile; PAH/worst-panel behavior and threshold variation add caution. V3 candidates are evidence only, not final replacements.

## Inference and reproduction

Use `python scripts/generate_final_predictions.py --input-csv INPUT.csv --output-csv outputs/final_predictions.csv --locked-artifact-dir artifacts/final_locked`.

Use `python scripts/verify_final_locked_model.py` to recompute metrics from locked prediction evidence.

No official hidden-test metric is claimed. Hidden-test distribution shift and lack of prospective clinical validation remain limitations.
"""
    (REPORTS / "FINAL_MODEL_DECISION.md").write_text(decision_md, encoding="utf-8")
    candidate = pd.read_csv(ROOT / "reports/v3/train_test/tables/model_selection_comparison.csv")
    et = candidate[candidate.candidate_id.eq("extratrees__v3_no_target_encoding")].sort_values("local_test_f1_macro", ascending=False).head(1)
    rejected = [{"candidate": "hist_gradient_boosting__v3_safe_minimal", "evaluation": "repeated contamination-aware stress validation", "master_f1_macro": stress["master_f1_macro_mean"], "master_mcc": stress["master_mcc_mean"], "panel_combined_f1_macro": stress["panel_f1_macro_mean"], "panel_combined_mcc": stress["panel_mcc_mean"], "decision": "reject_candidate", "reason": "MASTER F1/MCC below protected baseline; PAH/worst-panel weakness; threshold instability; protocol non-identical."}]
    if not et.empty:
        item = et.iloc[0]; rejected.append({"candidate": item.candidate_id, "evaluation": "local holdout", "master_f1_macro": item.local_test_f1_macro, "master_mcc": item.local_test_mcc, "panel_combined_f1_macro": item.panel_combined_f1_macro, "panel_combined_mcc": item.panel_combined_mcc, "decision": "reject_candidate", "reason": "Weaker than protected baseline and internal split is not replacement evidence."})
    rejected.append({"candidate": "all_other_v3_local_candidates", "evaluation": "local holdout", "master_f1_macro": None, "master_mcc": None, "panel_combined_f1_macro": None, "panel_combined_mcc": None, "decision": "exploratory/rejected", "reason": "No robust replacement gate passed; preserved as evidence only."})
    rejected_df = pd.DataFrame(rejected); rejected_df.to_csv(TABLES / "rejected_candidate_summary.csv", index=False)
    (REPORTS / "rejected_candidate_summary.md").write_text("# Rejected Candidate Summary\n\nStrong panel or KANSER behavior alone is not enough to replace the final model. MASTER decision metrics and robust gates remain decisive. V3 work is preserved as evidence but not promoted.\n\n" + rejected_df.to_markdown(index=False) + "\n", encoding="utf-8")
    summary = f"""# Final Project Summary

This TEKNOFEST 2026 Healthcare AI project predicts pathogenic versus benign missense variants from competition genomic tables (MASTER, KANSER, CFTR, PAH). It is a decision-support research model.

Leakage prevention excludes `Variant_ID` and labels from model features. Features include AL/frequency, EK/conservation, categorical metadata, amino-acid substitution, and missingness representations.

The final model locked is `lightgbm_conservative_regularized` at threshold 0.471, with no calibration and no ensemble replacement. MASTER OOF F1-macro/MCC are {oof['f1_macro']:.4f}/{oof['mcc']:.4f}; panel-unique F1-macro/MCC are {panel['f1_macro']:.4f}/{panel['mcc']:.4f}.

V3 HistGradientBoosting and all local-holdout candidates are rejected/exploratory evidence. Inference is ready through the locked CLI. No official hidden-test metric claimed; official labels are unavailable locally. Final submission recommendation: use the locked baseline only when an organizer-format unlabeled test CSV is supplied.
"""
    (REPORTS / "FINAL_PROJECT_SUMMARY.md").write_text(summary, encoding="utf-8")
    final_status = f"""# Final Status

- Final model: `lightgbm_conservative_regularized`
- Threshold: `0.471` (`profile_f1_macro_opt`)
- MASTER OOF: F1-macro {oof['f1_macro']:.4f}; MCC {oof['mcc']:.4f}
- Panel-unique: F1-macro {panel['f1_macro']:.4f}; MCC {panel['mcc']:.4f}
- Locked artifacts: `artifacts/final_locked/`
- Verify: `python scripts/verify_final_locked_model.py`
- Predict: `python scripts/generate_final_predictions.py --input-csv INPUT.csv --output-csv outputs/final_predictions.csv --locked-artifact-dir artifacts/final_locked`
- Official test status: no official test CSV/labels available locally; no official hidden-test metric claim.
- V3 HistGradientBoosting is rejected/exploratory; the protected baseline remains final.
"""
    (ROOT / "FINAL_STATUS.md").write_text(final_status, encoding="utf-8")
    (REPORTS / "final_inference_readiness.md").write_text("# Final Inference Readiness\n\n`generate_final_predictions.py` loads only locked model/preprocessing artifacts, retains Variant_ID as output metadata, ignores Label if present, aligns locked feature columns, and writes probability plus thresholded labels. No official test CSV exists locally, so no prediction file was generated. Metrics are unavailable without explicitly supplied labeled local data.\n", encoding="utf-8")
    pd.DataFrame([{"file_path": "none", "exists": False, "has_label_column": False, "action_taken": "no official test CSV found", "prediction_file_created": False, "prediction_file_path": "", "metrics_computed": False, "notes": "no official hidden-test metric is claimed"}]).to_csv(TABLES / "official_test_status.csv", index=False)
    print(f"Locked {len(rows)} artifacts; missing {len(missing)} expected artifacts.")


if __name__ == "__main__":
    main()
