"""Recompute protected metrics from locked prediction evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "artifacts" / "final_locked"
REPORTS = ROOT / "reports" / "final_model_lock"


def metrics(frame: pd.DataFrame) -> dict:
    y, p = frame.Label.astype(int), frame.score.astype(float)
    pred = (p >= 0.471).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "f1_macro": f1_score(y, pred, average="macro"), "mcc": matthews_corrcoef(y, pred), "tn": int(((y == 0) & (pred == 0)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()), "fn": int(((y == 1) & (pred == 0)).sum()), "tp": int(((y == 1) & (pred == 1)).sum())}


def main() -> None:
    decision = json.loads((LOCK / "metadata/final_model_decision.json").read_text(encoding="utf-8"))["source_decision"]
    threshold = json.loads((LOCK / "metadata/final_threshold.json").read_text(encoding="utf-8"))["threshold"]
    oof, panel = metrics(pd.read_csv(LOCK / "predictions/final_master_cv_predictions.csv")), metrics(pd.read_csv(LOCK / "predictions/final_panel_predictions.csv"))
    rows = []
    for split, actual, expected in (("MASTER_OOF", oof, decision["oof_metrics"]), ("panel_unique_combined", panel, decision["panel_unique_combined_metrics"])):
        row = {"split": split, "threshold": threshold, "threshold_match": np.isclose(threshold, 0.471), "confusion_match": all(actual[key] == expected[key] for key in ("tn", "fp", "fn", "tp"))}
        for key in ("roc_auc", "pr_auc", "f1_macro", "mcc"):
            row[f"recomputed_{key}"] = actual[key]; row[f"reported_{key}"] = expected[key]; row[f"abs_diff_{key}"] = abs(actual[key] - expected[key])
        row["status"] = "pass" if row["threshold_match"] and row["confusion_match"] and all(row[f"abs_diff_{key}"] <= 0.0005 for key in ("roc_auc", "pr_auc", "f1_macro", "mcc")) else "fail"
        rows.append(row)
    features = json.loads((LOCK / "metadata/final_feature_list.json").read_text(encoding="utf-8"))
    feature_names = features.get("feature_columns", features) if isinstance(features, dict) else features
    leakage_ok = not any("variant_id" in str(name).lower() or str(name).lower() == "label" for name in feature_names)
    table = pd.DataFrame(rows); table["feature_leakage_check"] = leakage_ok; table.to_csv(REPORTS / "tables/final_metric_reverification.csv", index=False)
    passed = bool(table.status.eq("pass").all() and leakage_ok)
    outcome = {"passed": bool(passed), "threshold": float(threshold), "feature_leakage_check": bool(leakage_ok), "official_metric_claimed": False, "results": json.loads(table.drop(columns=["feature_leakage_check"]).to_json(orient="records"))}
    (LOCK / "metadata/final_metric_reverification.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    (REPORTS / "final_metric_reverification.md").write_text("# Final Metric Reverification\n\n" + table.to_markdown(index=False) + "\n\nStatus: " + ("PASS" if passed else "FAIL") + ". Metrics were recomputed from locked prediction files; no official hidden-test metric is claimed.\n", encoding="utf-8")
    if not passed:
        raise SystemExit("Final locked metric reverification failed.")
    print("Final locked metric reverification passed.")


if __name__ == "__main__":
    main()
