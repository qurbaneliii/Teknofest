"""Regenerate derived V3 holdout status/report files from completed saved tables."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from teknofest_v3.data import find_unlabeled_test_files
from v3_train_test_evaluate import write_report


def main() -> None:
    reports = ROOT / "reports" / "v3" / "train_test"
    tables = reports / "tables"
    test_files = find_unlabeled_test_files(ROOT / "teknofest2026_artificialintelligenceinhealtcare-main")
    official = pd.DataFrame(
        [{
            "file_path": "; ".join(str(path) for path in test_files) if test_files else "none",
            "exists": bool(test_files),
            "has_label_column": False,
            "action_taken": "no official test CSV found" if not test_files else "no predictions generated because no candidate is final",
            "metrics_computed": False,
            "prediction_file_created": False,
            "notes": "no official test metric is claimed",
        }]
    )
    official.to_csv(tables / "official_test_prediction_status.csv", index=False)
    write_report(
        reports,
        pd.read_csv(tables / "train_test_split_summary.csv"),
        pd.read_csv(tables / "feature_set_comparison.csv"),
        pd.read_csv(tables / "model_availability.csv"),
        pd.read_csv(tables / "model_warnings.csv"),
        pd.read_csv(tables / "model_train_test_metrics.csv"),
        pd.read_csv(tables / "model_panel_metrics.csv"),
        pd.read_csv(tables / "model_selection_comparison.csv"),
        official,
    )
    print("V3 train/test report and official-test status regenerated from saved tables.")


if __name__ == "__main__":
    main()
