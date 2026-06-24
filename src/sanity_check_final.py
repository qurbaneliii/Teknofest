"""Fast integrity checks for the audited final competition package."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medical_metrics import compute_medical_metrics  # noqa: E402


def main() -> None:
    decision_path = ROOT / "artifacts" / "metrics" / "audited_final_model_decision.json"
    required = [decision_path, ROOT / "artifacts" / "predictions" / "audited_master_only_oof_predictions.csv", ROOT / "artifacts" / "predictions" / "audited_panel_unique_predictions.csv", ROOT / "outputs" / "final" / "final_metric_verification.csv", ROOT / "outputs" / "final" / "subgroup_metrics.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Missing final-package files: " + ", ".join(missing))
    decision = json.loads(decision_path.read_text(encoding="utf-8")); bundle = joblib.load(decision["artifact_path"])
    if {"model", "feature_engineer", "feature_columns"} - set(bundle): raise ValueError("Final model bundle is incomplete.")
    if any(c in bundle["feature_columns"] for c in ("Label", "Variant_ID")): raise ValueError("Target or identifier leaked into final model features.")
    oof = pd.read_csv(required[1]); verified = pd.read_csv(required[3]); threshold = float(decision["threshold"])
    recomputed = compute_medical_metrics(oof.Label, oof.score, threshold); row = verified.loc[verified.evaluation_split.eq("MASTER_ONLY_CV")].iloc[0]
    for column, value in {"f1_macro": recomputed["f1_macro"], "mcc": recomputed["mcc"], "roc_auc": recomputed["roc_auc"], "pr_auc": recomputed["pr_auc"]}.items():
        if abs(float(row[column]) - float(value)) > 1e-12: raise AssertionError(f"Metric mismatch for {column}")
    if "official_test" in " ".join(verified.evaluation_split.astype(str)).lower(): raise AssertionError("Official test metrics must not be claimed without labels.")
    print("Sanity check passed: final artifact, schema, no-ID/no-label features, and recomputed metrics agree.")


if __name__ == "__main__":
    main()
