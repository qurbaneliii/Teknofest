from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import discover_data_dir
from final_model_zoo import run_final_model_zoo
from teknofest.data_prep import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the leakage-safe TEKNOFEST final model zoo.")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    data_dir = args.data_dir or discover_data_dir(PROJECT_ROOT)
    result = run_final_model_zoo(prepare_data(data_dir))
    print(f"Model-zoo OOF predictions: {PROJECT_ROOT / 'artifacts' / 'predictions' / 'model_zoo_oof_predictions.csv'}")
    successful = result["metrics"][result["metrics"]["status"].eq("success")]
    print(f"Successful model-zoo candidates: {len(successful)}")
    failures = result["metrics"][result["metrics"]["status"].eq("failed")]
    if not failures.empty:
        print("Failed optional candidates:")
        for _, row in failures.iterrows():
            print(f"- {row['model_id']}: {row['failure_reason']}")


if __name__ == "__main__":
    main()
