from __future__ import annotations

import argparse
from pathlib import Path

from teknofest.data_prep import prepare_data
from teknofest.explainability import run_explainability


DEFAULT_DATA_DIR = Path("teknofest2026_artificialintelligenceinhealtcare-main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Part IV, Section F explainability.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/explainability"))
    parser.add_argument("--sample-size", type=int, default=0, help="Use 0 for full MASTER.")
    args = parser.parse_args()

    prepared = prepare_data(args.data_dir)
    run_explainability(
        prepared=prepared,
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        sample_size=None if args.sample_size <= 0 else args.sample_size,
    )
    print("Fifth section explainability complete.")
    print(f"Explainability outputs written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
