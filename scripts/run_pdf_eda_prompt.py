from __future__ import annotations

import argparse
from pathlib import Path

from teknofest.eda import run_eda


DEFAULT_DATA_DIR = Path("teknofest2026_artificialintelligenceinhealtcare-main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the EDA instructions from Adsiz dokuman.pdf.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()

    run_eda(args.data_dir, args.out_dir)
    print("PDF EDA prompt applied.")
    print(f"EDA report written to: {(args.out_dir / 'EDA_REPORT.md').resolve()}")
    print(f"Tables written to: {(args.out_dir / 'tables').resolve()}")
    print(f"Figures written to: {(args.out_dir / 'figures').resolve()}")


if __name__ == "__main__":
    main()
