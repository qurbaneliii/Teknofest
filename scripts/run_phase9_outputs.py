from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import discover_data_dir
from phase9_outputs import generate_phase9_outputs
from teknofest.data_prep import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PHASE 9.5 visualization and metric outputs.")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or discover_data_dir(PROJECT_ROOT)
    prepared = prepare_data(data_dir)
    generate_phase9_outputs(prepared)
    print("PHASE 9.5 outputs generated.")
    print(f"Figures: {(PROJECT_ROOT / 'reports' / 'figures').resolve()}")
    print(f"Tables: {(PROJECT_ROOT / 'reports' / 'tables').resolve()}")


if __name__ == "__main__":
    main()
