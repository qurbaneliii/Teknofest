"""Verify the final lock contract without training or modifying protected artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "artifacts" / "final_locked"
REPORTS = ROOT / "reports" / "final_model_lock"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        LOCK / "metadata/final_model_decision.json", LOCK / "metadata/final_artifact_manifest.json",
        LOCK / "metadata/final_metric_reverification.json", REPORTS / "FINAL_MODEL_DECISION.md",
        REPORTS / "FINAL_PROJECT_SUMMARY.md", REPORTS / "rejected_candidate_summary.md",
        REPORTS / "tables/official_test_status.csv", ROOT / "FINAL_STATUS.md",
        ROOT / "scripts/generate_final_predictions.py", ROOT / "reports/final_model_lock/tables/final_metric_reverification.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    failed: list[str] = []
    manifest = pd.read_csv(REPORTS / "tables/final_artifact_manifest.csv")
    for row in manifest.itertuples(index=False):
        original, locked = Path(row.original_path), Path(row.locked_copy_path)
        if not original.exists() or not locked.exists() or sha256(original) != sha256(locked):
            failed.append(f"hash mismatch: {row.artifact_name}")
    decision = json.loads((LOCK / "metadata/final_model_decision.json").read_text(encoding="utf-8"))
    verification = json.loads((LOCK / "metadata/final_metric_reverification.json").read_text(encoding="utf-8"))
    if decision["model_id"] != "lightgbm_conservative_regularized": failed.append("final model ID mismatch")
    if float(decision["threshold"]) != 0.471: failed.append("final threshold mismatch")
    if decision["calibration"] != "none": failed.append("calibration is not none")
    if bool(decision["ensemble_replaced_lightgbm"]): failed.append("ensemble replacement is enabled")
    if not verification.get("passed", False): failed.append("metric reverification did not pass")
    rejected = (REPORTS / "rejected_candidate_summary.md").read_text(encoding="utf-8") if (REPORTS / "rejected_candidate_summary.md").exists() else ""
    if "HistGradientBoosting" not in rejected and "hist_gradient_boosting" not in rejected: failed.append("HGB rejection evidence missing")
    pytest = subprocess.run([sys.executable, "-m", "pytest", "tests"], cwd=ROOT, text=True, capture_output=True)
    pytest_passed = pytest.returncode == 0
    if not pytest_passed: failed.append("pytest failed")
    status = {"final_contract_complete": not missing and not failed, "final_model_id": decision["model_id"], "final_threshold": decision["threshold"], "calibration": decision["calibration"], "ensemble_replaced_lightgbm": decision["ensemble_replaced_lightgbm"], "protected_baseline_overwritten": False, "official_metric_claimed": False, "hgb_promoted": False, "missing_files": missing, "failed_checks": failed, "warnings": ["No official test CSV was found locally."], "final_decision": "protected LightGBM baseline locked as final", "next_action": "Use the locked inference CLI only when organizer-format test data is supplied.", "pytest_passed": pytest_passed}
    (REPORTS / "final_contract_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (REPORTS / "final_contract_status.md").write_text("# Final Contract Status\n\n```json\n" + json.dumps(status, indent=2) + "\n```\n\nPytest output:\n\n```text\n" + pytest.stdout[-4000:] + "\n```\n", encoding="utf-8")
    print("Final contract complete." if status["final_contract_complete"] else "Final contract incomplete: " + "; ".join(failed + missing))
    if not status["final_contract_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
