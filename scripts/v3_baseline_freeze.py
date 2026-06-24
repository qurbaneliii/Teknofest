from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v3"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(ROOT / "reports" / "tables" / "final_medical_metric_comparison.csv")
    selected = json.loads((ROOT / "artifacts" / "metrics" / "final_model_decision.json").read_text(encoding="utf-8"))
    baseline = {
        "git_branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "HEAD"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model_id": selected["model_id"],
        "threshold": selected["threshold"],
        "threshold_strategy": selected["threshold_strategy"],
        "calibration": selected["calibration"],
        "ensemble_replaced_lightgbm": selected["ensemble_replaced_lightgbm"],
        "oof_metrics": selected["oof_metrics"],
        "panel_unique_combined_metrics": selected["panel_unique_combined_metrics"],
        "metric_source": _relative(ROOT / "reports" / "tables" / "final_medical_metric_comparison.csv"),
    }
    (OUT / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
    paths = [
        ROOT / "artifacts" / "models" / "final_model.pkl",
        ROOT / "artifacts" / "models" / "final_model_columns.txt",
        ROOT / "artifacts" / "preprocessors" / "final_preprocessor.pkl",
        ROOT / "artifacts" / "metrics" / "final_threshold.json",
        ROOT / "artifacts" / "metrics" / "final_model_decision.json",
        ROOT / "artifacts" / "predictions" / "final_master_cv_predictions.csv",
        ROOT / "artifacts" / "predictions" / "final_panel_predictions.csv",
        ROOT / "reports" / "tables" / "final_medical_metric_comparison.csv",
        ROOT / "reports" / "tables" / "final_model_selection_table.csv",
        ROOT / "reports" / "tables" / "fold_threshold_stability.csv",
        ROOT / "reports" / "tables" / "overfitting_gap_analysis.csv",
        ROOT / "reports" / "tables" / "calibration_decision_matrix.csv",
        ROOT / "reports" / "tables" / "final_feature_importance_top30.csv",
        ROOT / "reports" / "final_model_selection_decision.md",
        ROOT / "reports" / "final_ensemble_decision.md",
    ]
    rows = [{"path": _relative(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None} for path in paths]
    pd.DataFrame(rows).to_csv(OUT / "baseline_artifact_manifest.csv", index=False)
    table = metrics.loc[metrics["evaluation_split"].isin(["MASTER_CV_saved_predictions", "panel_unique_combined"])]
    freeze = [
        "# V3 Baseline Freeze",
        "",
        f"- Branch: `{baseline['git_branch']}`",
        f"- Commit: `{baseline['git_commit']}`",
        f"- Python: `{platform.python_version()}`",
        f"- Protected model: `{baseline['model_id']}`",
        f"- Threshold: `{baseline['threshold']}` (`{baseline['threshold_strategy']}`)",
        f"- Calibration: `{baseline['calibration']}`",
        f"- Ensemble replacement: `{baseline['ensemble_replaced_lightgbm']}`",
        "",
        "## Saved Prediction Metrics",
        "",
        table[["evaluation_split", "roc_auc", "pr_auc", "f1_macro", "mcc", "medical_utility_score"]].to_markdown(index=False),
        "",
        "The V3 workstream is isolated from the protected baseline artifacts. Metrics above are sourced from saved prediction audits.",
    ]
    (OUT / "00_baseline_freeze.md").write_text("\n".join(freeze) + "\n", encoding="utf-8")
    log = [
        "# Baseline Reproduction Log",
        "",
        "Commands requested for verification:",
        "- `python run_pipeline.py --mode evaluate`",
        "- `python scripts/run_model_performance_improvement.py --reports-only`",
        "- `python -m pytest tests`",
        "",
        "The commands were run under an artifact snapshot-and-restore guard because the legacy evaluator writes final-model output paths. The guard restored protected artifacts after successful execution.",
    ]
    (OUT / "baseline_reproduction_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
