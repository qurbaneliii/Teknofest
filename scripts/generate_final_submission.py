from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from final_inference import generate_final_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a TEKNOFEST final-model submission CSV without using labels.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Organizer data directory, used when --input-csv is omitted.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Raw organizer-format inference CSV.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "predictions" / "final_submission_predictions.csv")
    args = parser.parse_args()
    if args.input_csv is not None:
        input_path = args.input_csv
    elif args.data_dir is not None:
        input_path = args.data_dir / "YARISMA_TRAIN_MASTER.csv"
    else:
        candidates = list(PROJECT_ROOT.rglob("YARISMA_TRAIN_MASTER.csv"))
        if not candidates:
            raise FileNotFoundError("Provide --input-csv or --data-dir containing organizer-format CSV data.")
        input_path = candidates[0]
    output, destination = generate_final_submission(pd.read_csv(input_path), args.output)
    print(f"Submission written to: {destination.resolve()}")
    print(f"Rows: {len(output)}; source labels, if present, were ignored.")


if __name__ == "__main__":
    main()
