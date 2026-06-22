from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import discover_data_dir
from final_competition_pipeline import run_final_competition_pipeline
from teknofest.data_prep import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete leakage-safe final competition workflow.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--skip-repeated-validation", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir or discover_data_dir(PROJECT_ROOT)
    result = run_final_competition_pipeline(prepare_data(data_dir), not args.skip_repeated_validation)
    print(f"Selected candidate: {result['selected']['candidate_id']}")
    print(f"Submission: {result['submission_path']}")
    print(f"Report assets: {result['report_assets']}")


if __name__ == "__main__":
    main()
