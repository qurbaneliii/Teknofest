from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import discover_data_dir
from final_report_assets import generate_final_report_assets
from teknofest.data_prep import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TEKNOFEST report-ready final model assets.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "final_report_assets")
    args = parser.parse_args()
    data_dir = args.data_dir or discover_data_dir(PROJECT_ROOT)
    destination = generate_final_report_assets(prepare_data(data_dir), args.output_dir)
    print(f"Report assets written to: {destination.resolve()}")


if __name__ == "__main__":
    main()
