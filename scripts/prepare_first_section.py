from __future__ import annotations

import argparse
from pathlib import Path

from teknofest.data_prep import overlap_summary, prepare_data, validate_first_section


DEFAULT_DATA_DIR = Path("teknofest2026_artificialintelligenceinhealtcare-main")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Part IV, Section B data preparation from the master prompt."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    prepared = prepare_data(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = overlap_summary(prepared)
    summary.to_csv(args.out_dir / "first_section_overlap_summary.csv", index=False)

    for name in ("master", "kanser", "pah", "cftr"):
        getattr(prepared, name).to_csv(args.out_dir / f"{name}_prepared.csv", index=False)

    prepared.kanser_unique.to_csv(args.out_dir / "kanser_unique.csv", index=False)
    prepared.pah_unique.to_csv(args.out_dir / "pah_unique.csv", index=False)
    prepared.cftr_unique.to_csv(args.out_dir / "cftr_unique.csv", index=False)

    validation = validate_first_section(prepared)
    print("First section data preparation complete.")
    print(summary.to_string(index=False))
    print(f"AL_185 dropped: {validation['al_185_dropped']}")
    print(f"AL_186 kept: {validation['al_186_kept']}")
    print(f"AL columns after drop: {validation['n_al_cols_after_drop']}")
    print(f"Processed files written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
