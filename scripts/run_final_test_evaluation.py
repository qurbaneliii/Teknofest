from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from final_test_evaluation import REPORT_DIR, run_final_test_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final protected-model test discovery and evaluation.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "teknofest2026_artificialintelligenceinhealtcare-main")
    args = parser.parse_args()
    result = run_final_test_evaluation(args.data_dir)
    report = PROJECT_ROOT / "reports" / "final_test_performance_report.md"
    if result["status"] == "no_test_files":
        text = """# Final Test Performance Report

## Executive Summary

No official test CSV was found, so the protected model was not run on training data as a substitute. Official test labels are not available, so true accuracy, F1, MCC, ROC-AUC, PR-AUC, sensitivity, and specificity cannot be computed on the official test set. The report therefore provides inference validity, confidence analysis, schema checks, distribution-shift analysis, and validation/panel-based performance evidence when an official test file is provided.

## Final Model

`lightgbm_conservative_regularized`, threshold `0.471`, strategy `profile_f1_macro_opt`, calibration `none`; the ensemble did not replace LightGBM.

## Validation Recap

Saved MASTER CV: ROC-AUC 0.8475, PR-AUC 0.9025, F1-macro 0.7764, MCC 0.5548, MedicalUtilityScore 0.7747. Saved panel-unique combined: F1-macro 0.7708 and MCC 0.5825.

## Conclusion

The model's true official test performance cannot be measured locally because test labels are unavailable. Based on saved validation and panel-unique evaluations, the model is moderate-to-good, leakage-aware, and stable enough for competition submission, but hidden-test performance remains uncertain.

See `reports/test_evaluation/test_data_discovery.md` for expected file names and locations.
"""
    else:
        text = f"# Final Test Performance Report\n\nEvaluation completed in `{result['status']}` mode for `{result['predictions']}` rows. See `reports/test_evaluation/` for generated metrics, plots, and audits.\n"
    report.write_text(text, encoding="utf-8")
    print(result["status"])
    print(f"Discovery report: {(REPORT_DIR / 'test_data_discovery.md').resolve()}")
    print(f"Final report: {report.resolve()}")


if __name__ == "__main__":
    main()
