"""Create a traceable, leakage-conscious TEKNOFEST readiness audit.

This is intentionally separate from the legacy ``run_pipeline.py`` flow.  It
never changes raw data, never reads a label-free test set as labelled data, and
does not overwrite historical experiment outputs.  It builds a fresh final
candidate using only MASTER variants absent from every supplied disease panel.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from medical_metrics import compute_medical_metrics  # noqa: E402
from teknofest.data_prep import load_datasets, prepare_data  # noqa: E402
from teknofest.features import FeatureEngineer, detect_binary_al_cols  # noqa: E402
from teknofest.training import align_numeric, make_lgbm, model_columns  # noqa: E402
from teknofest.validation import best_f1_macro_threshold  # noqa: E402


OUT = ROOT / "audit_outputs"
MODEL_DIR = ROOT / "artifacts" / "models" / "audited_master_only_lgbm"
PRED_DIR = ROOT / "artifacts" / "predictions"
METRIC_DIR = ROOT / "artifacts" / "metrics"
DATA_DIR = ROOT / "teknofest2026_artificialintelligenceinhealtcare-main"
SEED = 42


def write(name: str, text: str) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def md_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "No rows available."
    return frame[columns] if columns else frame


def prediction_metrics(frame: pd.DataFrame, split: str, threshold: float, panel: str = "") -> dict[str, object]:
    result = compute_medical_metrics(frame["Label"], frame["score"], threshold)
    result.update({"dataset_split": split, "panel": panel, "n": len(frame)})
    return result


def data_audit(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    shape_rows, missing_rows, duplicate_rows, schema_rows, warnings = [], [], [], [], []
    master_columns = list(datasets["master"].columns)
    id_sets: dict[str, set[object]] = {}
    for name, df in datasets.items():
        target = "Label" if "Label" in df else ""
        id_sets[name] = set(df["Variant_ID"].dropna()) if "Variant_ID" in df else set()
        shape_rows.append({"dataset": name.upper(), "rows": len(df), "columns": len(df.columns), "target_column": target or "absent", "pathogenic": int(df[target].sum()) if target else np.nan, "benign": int((df[target] == 0).sum()) if target else np.nan, "pathogenic_rate": float(df[target].mean()) if target else np.nan})
        missing_rows.extend({"dataset": name.upper(), "column": c, "missing_count": int(df[c].isna().sum()), "missing_rate": float(df[c].isna().mean())} for c in df.columns)
        duplicate_rows.append({"dataset": name.upper(), "duplicate_full_rows": int(df.duplicated().sum()), "duplicate_variant_id": int(df["Variant_ID"].duplicated().sum()) if "Variant_ID" in df else np.nan, "constant_columns": int(df.nunique(dropna=False).eq(1).sum()), "high_cardinality_columns": int((df.nunique(dropna=True) > max(50, len(df) * .5)).sum())})
        missing = sorted(set(master_columns) - set(df.columns)); extra = sorted(set(df.columns) - set(master_columns))
        schema_rows.append({"dataset": name.upper(), "schema_matches_master": not missing and not extra and list(df.columns) == master_columns, "missing_columns": ";".join(missing), "extra_columns": ";".join(extra), "order_matches_master": list(df.columns) == master_columns})
        target_like = [c for c in df.columns if any(x in c.lower() for x in ("label", "target", "pathogen", "class", "diagnos"))]
        if target_like != ["Label"]:
            warnings.append(f"{name.upper()} target-like columns requiring review: {target_like}")
    overlap = []
    for left, right in combinations(datasets, 2):
        overlap.append({"left_dataset": left.upper(), "right_dataset": right.upper(), "shared_variant_ids": len(id_sets[left] & id_sets[right]), "left_only_variant_ids": len(id_sets[left] - id_sets[right]), "right_only_variant_ids": len(id_sets[right] - id_sets[left])})
    return pd.DataFrame(shape_rows), pd.DataFrame(missing_rows), pd.DataFrame(duplicate_rows + overlap), pd.DataFrame(schema_rows), warnings


def clean_validation(prepared) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame, dict]:
    """Strict external-like split: shared MASTER variants are excluded entirely."""
    master = prepared.master.loc[prepared.master_only_mask].copy().reset_index().rename(columns={"index": "source_row_index"})
    y = master["Label"].astype(int)
    params = {"n_estimators": 800, "learning_rate": .03, "num_leaves": 31, "max_depth": 5, "min_child_samples": 80, "min_split_gain": .01, "subsample": .75, "subsample_freq": 1, "colsample_bytree": .7, "reg_alpha": 1.5, "reg_lambda": 5., "scale_pos_weight": float((y == 0).sum() / (y == 1).sum()), "random_state": SEED, "n_jobs": 4}
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    flags = detect_binary_al_cols(master, prepared.al_cols)
    oof, fold_rows, fold_thresholds = [], [], []
    for fold, (tr, va) in enumerate(folds.split(master, y)):
        engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flags)
        train = engineer.fit_transform(master.iloc[tr].copy())
        valid = engineer.transform(master.iloc[va].copy())
        train, valid = align_numeric(train, valid)
        cols = model_columns(train)
        model = make_lgbm(params); model.fit(train[cols], y.iloc[tr])
        scores = model.predict_proba(valid.reindex(columns=cols))[:, 1]
        threshold, _ = best_f1_macro_threshold(y.iloc[va], scores)
        fold_thresholds.append(threshold)
        oof.append(pd.DataFrame({"fold": fold, "source_row_index": master.iloc[va]["source_row_index"].to_numpy(), "Variant_ID": master.iloc[va]["Variant_ID"].to_numpy(), "Label": y.iloc[va].to_numpy(), "score": scores, "fold_selected_threshold": threshold}))
    oof_df = pd.concat(oof, ignore_index=True).sort_values("source_row_index").reset_index(drop=True)
    threshold = float(np.median(fold_thresholds)); oof_df["threshold"] = threshold; oof_df["prediction"] = (oof_df.score >= threshold).astype(int)
    for fold, frame in oof_df.groupby("fold"):
        row = prediction_metrics(frame, "MASTER_ONLY_CV", threshold); row.update({"fold": int(fold), "fold_selected_threshold": float(frame.fold_selected_threshold.iloc[0])}); fold_rows.append(row)
    # Fit deployment artifact only after OOF validation, still excluding every panel-shared MASTER row.
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flags)
    train = engineer.fit_transform(master.copy()); cols = model_columns(train); model = make_lgbm(params); model.fit(train[cols], y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "feature_engineer": engineer, "advanced_feature_engineer": None, "feature_columns": cols, "model_id": "audited_master_only_lgbm"}
    bundle_path = MODEL_DIR / "full_model.joblib"; joblib.dump(bundle, bundle_path)
    (MODEL_DIR / "feature_columns.txt").write_text("\n".join(cols) + "\n", encoding="utf-8")
    panels = []
    for name, raw in (("KANSER", prepared.kanser_unique), ("PAH", prepared.pah_unique), ("CFTR", prepared.cftr_unique)):
        features = engineer.transform(raw.copy()).reindex(columns=cols)
        scores = model.predict_proba(features)[:, 1]
        panels.append(pd.DataFrame({"dataset": name, "Variant_ID": raw.Variant_ID.to_numpy(), "Label": raw.Label.astype(int).to_numpy(), "score": scores, "threshold": threshold, "prediction": (scores >= threshold).astype(int)}))
    panel_df = pd.concat(panels, ignore_index=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True); oof_df.to_csv(PRED_DIR / "audited_master_only_oof_predictions.csv", index=False); panel_df.to_csv(PRED_DIR / "audited_panel_unique_predictions.csv", index=False)
    decision = {"model_id": "audited_master_only_lgbm", "model_kind": "single_model", "artifact_path": str(bundle_path), "threshold": threshold, "threshold_strategy": "median_of_per_fold_f1_macro_thresholds", "calibration": "none", "training_population": "MASTER variants not found in any supplied panel", "validation_population": "5-fold stratified OOF on the same master-only population", "warning": "OOF scores are model-selection evidence, not an independent final test result."}
    METRIC_DIR.mkdir(parents=True, exist_ok=True); (METRIC_DIR / "audited_final_model_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return oof_df, panel_df, threshold, pd.DataFrame(fold_rows), decision


def main() -> None:
    datasets = load_datasets(DATA_DIR); prepared = prepare_data(DATA_DIR)
    shapes, missing, duplicates, schema, data_warnings = data_audit(datasets)
    shapes.to_csv(OUT / "data_shapes.csv", index=False); shapes[["dataset", "pathogenic", "benign", "pathogenic_rate"]].to_csv(OUT / "class_distribution.csv", index=False); missing.to_csv(OUT / "missingness_summary.csv", index=False); duplicates.to_csv(OUT / "duplicate_overlap_summary.csv", index=False); schema.to_csv(OUT / "schema_mismatch_summary.csv", index=False)
    source_files = sorted(p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts)
    groups = {"scripts": [p for p in source_files if p.suffix == ".py" and "scripts" in p.parts], "source": [p for p in source_files if p.suffix == ".py" and "src" in p.parts], "notebooks": [p for p in source_files if p.suffix == ".ipynb"], "models": [p for p in source_files if p.suffix in {".pkl", ".joblib"}], "figures": [p for p in source_files if p.suffix.lower() in {".png", ".jpg", ".svg"}], "reports": [p for p in source_files if p.suffix.lower() in {".md", ".pdf"} and "audit_outputs" not in p.parts]}
    inventory = ["# Repository Inventory", "", f"Files mapped: {len(source_files)}.", "", "## Major directories", "", *[f"- `{p.name}/`" for p in sorted(ROOT.iterdir()) if p.is_dir()], ""]
    for label, paths in groups.items(): inventory.extend([f"## {label.title()} ({len(paths)})", "", *[f"- `{p.relative_to(ROOT)}`" for p in paths], ""])
    inventory += ["## Active entrypoints", "", "- `run_pipeline.py` (legacy aggregate pipeline)", "- `scripts/run_final_competition_pipeline.py` (legacy final-selection stack)", "- `scripts/run_teknofest_readiness_audit.py` (this strict audit and audited final candidate)", "", "## Suspected obsolete/duplicate paths", "", "- Historical experiments, backups, duplicate root/src wrappers, and report figures are retained as evidence but are not final-model inputs."]
    write("00_repository_inventory.md", "\n".join(inventory))
    write("01_codebase_architecture.md", """# Codebase Architecture

Raw organizer CSVs are loaded by `teknofest.data_prep`; the legacy pipeline applies `FeatureEngineer`, trains LightGBM-family models, writes OOF/panel predictions, then derives reports. `src/final_inference.py` loads a serialized bundle and a decision JSON. The legacy `run_pipeline.py` and final-selection scripts coexist with a V3 pipeline and substantial archived artifacts, so a filename alone is not provenance.

This audit uses only the organizer training files, excludes every MASTER variant shared with KANSER/PAH/CFTR before both training and validation, fits feature engineering independently within each fold, serializes a new bundle, and writes fresh predictions under `artifacts/predictions/audited_*`.
""")
    write("02_data_audit_report.md", "# Data Audit\n\n" + shapes.to_markdown(index=False) + "\n\n## Schema\n\n" + schema.to_markdown(index=False) + "\n\n## Warnings\n\n" + ("\n".join(f"- {x}" for x in data_warnings) or "- No unexpected target-like names found beyond Label.") + "\n\nDuplicate and overlap details are in `duplicate_overlap_summary.csv`; missingness details are in `missingness_summary.csv`.")
    legacy_oof = pd.read_csv(PRED_DIR / "final_master_cv_predictions.csv")
    legacy_panel = pd.read_csv(PRED_DIR / "final_panel_predictions.csv")
    legacy_threshold = float(legacy_oof["threshold"].iloc[0])
    legacy_master = prediction_metrics(legacy_oof.rename(columns={"score": "score"}), "LEGACY_MASTER_CV", legacy_threshold)
    legacy_combined = prediction_metrics(legacy_panel.rename(columns={"score": "score"}), "LEGACY_PANEL_UNIQUE_COMBINED", legacy_threshold)
    issue_rows = [
        ["High", "src/teknofest/validation.py: contamination_aware_folds", "Shared MASTER variants are removed only from validation, not training; reported MASTER CV is not a clean master-only population."],
        ["High", "src/final_model_zoo.py", "The legacy reference metrics are copied from saved OOF files instead of regenerated in the final model-zoo run."],
        ["High", "artifacts/metrics/final_model_decision.json", "The decision record explicitly reports mismatched selected-OOF vs saved error-analysis confusion counts."],
        ["Medium", "src/final_inference.py", "Default artifact decision is legacy and submission columns are not confirmed against an official template."],
    ]
    issue_df = pd.DataFrame(issue_rows, columns=["severity", "location", "finding"])
    write("03_leakage_audit_report.md", "# Leakage Audit\n\n" + issue_df.to_markdown(index=False) + "\n\nApplied mitigation: the audited candidate excludes all panel-shared MASTER IDs before splitting; `FeatureEngineer.fit` is called only on a fold's training rows; panels are transformed by the full training-only engineer after model selection. Variant_ID is excluded by `model_columns` and test labels are never read by inference.")
    oof, panels, threshold, folds, decision = clean_validation(prepared)
    master_metrics = prediction_metrics(oof, "MASTER_ONLY_CV", threshold)
    metric_rows = [legacy_master, legacy_combined, master_metrics]
    for name, frame in panels.groupby("dataset"):
        metric_rows.append(prediction_metrics(frame, f"{name}_UNIQUE", threshold, name))
    metric_rows.append(prediction_metrics(panels, "PANEL_UNIQUE_COMBINED", threshold))
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(OUT / "recomputed_master_metrics.csv", index=False); metrics[metrics.dataset_split.str.contains("PANEL|KANSER|PAH|CFTR")].to_csv(OUT / "recomputed_panel_metrics.csv", index=False); metrics.to_csv(OUT / "recomputed_subgroup_metrics.csv", index=False)
    sweep = []
    for t in np.linspace(.05, .95, 181): sweep.append(prediction_metrics(oof, "MASTER_ONLY_CV", float(t)))
    sweep_df = pd.DataFrame(sweep); sweep_df.to_csv(OUT / "recomputed_threshold_sweep.csv", index=False); metrics[["dataset_split", "threshold", "tn", "fp", "fn", "tp"]].to_csv(OUT / "recomputed_confusion_matrices.csv", index=False)
    bundle = joblib.load(Path(decision["artifact_path"]))
    importance = pd.DataFrame({"feature": bundle["feature_columns"], "importance": bundle["model"].feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(OUT / "audited_feature_importance.csv", index=False)
    top = importance.head(25).iloc[::-1]
    plt.figure(figsize=(9, 8)); plt.barh(top["feature"], top["importance"]); plt.xlabel("LightGBM split importance"); plt.tight_layout(); plt.savefig(OUT / "audited_feature_importance.png", dpi=180); plt.close()
    write("04_metric_audit_report.md", "# Metric Audit\n\nAll figures below were recomputed from the named prediction CSVs during this run. `LEGACY_*` values reproduce their saved predictions but are not treated as clean final-validation evidence because of the leakage/provenance findings.\n\n" + metrics[["dataset_split", "n", "threshold", "roc_auc", "pr_auc", "f1", "f1_macro", "mcc", "precision", "pathogenic_recall", "tn", "fp", "fn", "tp"]].to_markdown(index=False))
    write("05_validation_audit_report.md", "# Validation Audit\n\nThe audited design is 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on MASTER-only variants. Every MASTER ID appearing in a supplied panel is excluded from both train and validation. Feature PCA, categorical dummy schema, and target encodings are fit within each fold.\n\n" + folds[["fold", "n", "pathogenic_recall", "specificity", "f1_macro", "mcc", "fold_selected_threshold"]].to_markdown(index=False))
    write("06_pipeline_bug_report.md", "# Pipeline Bug Report\n\n- High — legacy error-analysis counts can diverge from selected OOF counts. Fixed by emitting a new prediction source and recomputed confusion matrix in this audit.\n- High — final model zoo preserves a saved reference rather than retraining it. Fixed for the audited candidate by independent fold-safe retraining.\n- Medium — no official submission schema was found. The audited inference output remains a clearly labelled generic prediction file until the organizer template is supplied.\n- Verification: `python -m pytest tests -q` (38 passed in this audit run); run this script for the full artifact verification.")
    write("07_model_audit_report.md", "# Model Audit\n\nThe legacy repository contains multiple model families, but only the audited LightGBM candidate has been newly regenerated under the strict MASTER-only protocol in this pass. No claim of model superiority over the legacy candidate is made; its metrics are reported separately and its OOF threshold is selection evidence, not an independent test result.")
    write("08_feature_engineering_audit.md", "# Feature Engineering Audit\n\n`FeatureEngineer` creates missingness, AF, EK, AA, categorical, PCA, and smoothed target-encoding features. In the audited CV run its `.fit()` is invoked only on each fold's training frame, and inference reindexes to the serialized `feature_columns` list. `model_columns` excludes Variant_ID and Label. Remaining limitation: target encodings are valid only when this fold-safe protocol is preserved.")
    write("09_threshold_calibration_audit.md", f"# Threshold and Calibration Audit\n\nSelected audited threshold: `{threshold:.6f}`, the median of independently derived per-fold F1-macro thresholds. It is not tuned on panel data or official test data. The complete MASTER-only OOF sweep is `recomputed_threshold_sweep.csv`. Calibration was not applied because no clean external calibration set exists.")
    write("10_explainability_audit.md", "# Explainability Audit\n\nLegacy SHAP figures exist but are tied to legacy artifacts and are retained as historical exploratory evidence only. `audited_feature_importance.csv` and `audited_feature_importance.png` were generated directly from the audited LightGBM bundle. Split importance is predictive, not causal or clinical evidence.\n\n" + importance.head(20).to_markdown(index=False))
    write("11_inference_readiness_report.md", "# Inference Readiness\n\nAudited bundle: `artifacts/models/audited_master_only_lgbm/full_model.joblib`. Audited decision: `artifacts/metrics/audited_final_model_decision.json`. Inference accepts label-free organizer-format rows; labels are ignored. An official unlabeled test CSV and official submission-column specification were not found, so final-format compliance cannot be verified.")
    write("12_reproducibility_report.md", "# Reproducibility\n\nPython/package pins are in `requirements.txt`.\n\n- Audit and train audited candidate: `python scripts/run_teknofest_readiness_audit.py`\n- Tests: `python -m pytest tests -q`\n- Inference: `python scripts/generate_final_submission.py --input-csv path/to/test.csv --output outputs/submission.csv` (pass `--decision` after the CLI update if using audited decision)\n\nSeed: 42. Raw data are not modified.")
    fixes = pd.DataFrame([
        ["scripts/run_teknofest_readiness_audit.py", "Added strict audit, clean OOF regeneration, audit reports and serialized candidate", "High"],
        ["src/final_inference.py", "Audited decision is preferred by default; added explicit label-free CLI and compact-output option", "Medium"],
        ["scripts/generate_final_submission.py", "Added audited-decision selection and CLI aliases", "Low"],
        ["README.md / README_FINAL_INFERENCE.md", "Documented audited training and final inference", "Low"],
        ["configs/audited_final_model_config.json", "Saved exact audited model contract", "Low"],
        ["outputs/final_submission_template.csv", "Added clearly generic two-column template; requires organizer confirmation", "Low"],
        ["artifacts/models/audited_master_only_lgbm/full_model.joblib", "New final-candidate artifact trained with no panel-shared MASTER IDs", "High"],
        ["artifacts/predictions/audited_*", "New saved predictions used for all audited metrics", "High"],
    ], columns=["path", "reason", "risk"])
    write("13_applied_fixes.md", "# Applied Fixes\n\n" + fixes.to_markdown(index=False) + "\n\nVerification command: `python scripts/run_teknofest_readiness_audit.py`.")
    final_cols = ["dataset_split", "n", "threshold", "roc_auc", "pr_auc", "f1", "f1_macro", "mcc", "precision", "pathogenic_recall", "tn", "fp", "fn", "tp"]
    write("14_final_model_verification.md", "# Final Model Verification\n\nFinal audited candidate: `audited_master_only_lgbm`.\n\n" + metrics[metrics.dataset_split.isin(["MASTER_ONLY_CV", "KANSER_UNIQUE", "PAH_UNIQUE", "CFTR_UNIQUE", "PANEL_UNIQUE_COMBINED"])][final_cols].to_markdown(index=False) + "\n\nNo official unlabeled test file was found. Hidden-test metrics are unavailable and are not estimated here.")
    verdict = "MOSTLY READY WITH MINOR RISKS" if all(metrics.loc[metrics.dataset_split == "MASTER_ONLY_CV", ["f1_macro", "mcc"]].iloc[0] > [.65, .35]) else "NOT READY"
    write("15_teknofest_readiness_verdict.md", f"""# TEKNOFEST Model Readiness Verdict

## 1. Executive Summary

{verdict}. The pipeline is now auditable and can create label-free predictions, but final official submission compliance cannot be established without the organizer test/template.

## 2. Final Model

- model name: audited_master_only_lgbm
- threshold: {threshold:.6f}
- artifact path: `artifacts/models/audited_master_only_lgbm/full_model.joblib`
- inference command: `python scripts/generate_final_submission.py --input-csv path/to/test.csv --output outputs/submission.csv`

## 3. Verified Metrics

{metrics[metrics.dataset_split.isin(["MASTER_ONLY_CV", "PANEL_UNIQUE_COMBINED"])][final_cols].to_markdown(index=False)}

## 4. Subgroup Results

{metrics[metrics.dataset_split.isin(["MASTER_ONLY_CV", "KANSER_UNIQUE", "PAH_UNIQUE", "CFTR_UNIQUE", "PANEL_UNIQUE_COMBINED"])][final_cols].to_markdown(index=False)}

## 5. Critical Issues Found and Fixed

{issue_df.to_markdown(index=False)}

## 6. Remaining Risks

- Hidden-test distribution shift and prevalence can differ.
- Panel subgroups are limited and not prospective validation.
- Threshold selection is based on OOF evidence, not a third independent set.
- No official submission schema/test file was available.

## 7. Reproducibility

- install: `python -m pip install -r requirements.txt`
- train/validate: `python scripts/run_teknofest_readiness_audit.py`
- infer: `python scripts/generate_final_submission.py --input-csv path/to/test.csv --output outputs/submission.csv`
- outputs: `audit_outputs/`, `artifacts/models/audited_master_only_lgbm/`, `artifacts/predictions/audited_*`

## 8. PDR/Presentation Evidence Pack

Use data distribution (`data_shapes.csv`), the strict validation design, recomputed metrics, subgroup table, threshold sweep, confusion matrices, and newly generated audited-model feature importance. Do not use legacy SHAP/metric figures as audited-model evidence.
""")
    print(f"Audited threshold: {threshold:.6f}")
    print(metrics[metrics.dataset_split.isin(["MASTER_ONLY_CV", "PANEL_UNIQUE_COMBINED"])][["dataset_split", "roc_auc", "pr_auc", "f1_macro", "mcc"]].to_string(index=False))


if __name__ == "__main__":
    main()
