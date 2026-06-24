from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v2"
SKIP = {".git"}


def classify(path: Path) -> tuple[str, str]:
    relative = path.relative_to(ROOT).as_posix()
    name = path.name.lower()
    if any(part == "__pycache__" for part in path.parts) or name.endswith(".pyc") or relative.startswith(".codex_backups/"):
        return "SAFE_DELETE_CANDIDATE", "cache or local restoration backup; never delete before confirmation"
    if relative.startswith("teknofest2026_artificialintelligenceinhealtcare-main/") or name in {"readme.md", "requirements.txt"} or relative.startswith("tests/"):
        return "PROTECTED_DO_NOT_TOUCH", "raw data, reproducibility documentation, or tests"
    if relative.startswith("artifacts/models/final_") or relative.startswith("artifacts/preprocessors/final_") or relative.startswith("artifacts/predictions/final_") or relative.startswith("artifacts/metrics/final_"):
        return "PROTECTED_DO_NOT_TOUCH", "protected final artifact"
    if relative.startswith("src/teknofest/") or relative in {"run_pipeline.py", "scripts/run_model_performance_improvement.py"} or relative.startswith("src/final_"):
        return "CORE_V2_KEEP", "active protected pipeline dependency"
    if path.suffix in {".ipynb", ".png", ".jpg", ".svg"} or relative.startswith(("reports/eda/", "artifacts/metrics/experiments/", "reports/master_prompt/")):
        return "LEGACY_ARCHIVE", "useful previous evidence or generated visualization"
    if relative.startswith(("reports/", "artifacts/")):
        return "LEGACY_ARCHIVE", "report or experiment output not required for direct inference"
    if path.suffix == ".py":
        return "CORE_V2_KEEP", "script retained pending dependency audit"
    return "LEGACY_ARCHIVE", "retain as evidence pending archive execution"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.parts):
            continue
        group, reason = classify(path)
        rows.append({"path": path.relative_to(ROOT).as_posix(), "extension": path.suffix.lower(), "bytes": path.stat().st_size, "classification": group, "reason": reason})
    table = pd.DataFrame(rows).sort_values("path")
    table.to_csv(OUT / "repository_file_classification.csv", index=False)
    archive = table[table.classification.eq("LEGACY_ARCHIVE")].copy()
    archive["proposed_archive_root"] = "archive/previous_results/"
    archive.to_csv(OUT / "archive_plan.csv", index=False)
    summary = table.classification.value_counts().to_dict()
    (OUT / "repository_cleanup_audit.md").write_text(
        "# Repository Cleanup Audit\n\n"
        + "This is a read-only classification. No move or deletion was performed.\n\n"
        + "## Counts\n\n"
        + pd.DataFrame([summary]).to_markdown(index=False)
        + "\n\n## Archive Layout\n\n`archive/previous_results/{phase1_eda,phase2_feature_engineering,phase3_validation,phase4_training,phase5_explainability,phase9_outputs,phase10_final_selection,rejected_models,old_notebooks,old_reports,old_figures,old_optuna,misc}`\n",
        encoding="utf-8",
    )
    (OUT / "archive_plan.md").write_text(
        "# Archive Plan\n\nUseful legacy outputs will be moved only after human review. Protected final artifacts, raw data, official documents, tests, README, and requirements are excluded. Cache files and local backup directories are the only safe-delete candidates, and are not deleted by this phase.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
