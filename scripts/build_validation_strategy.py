from __future__ import annotations

import argparse
from pathlib import Path

from teknofest.data_prep import prepare_data
from teknofest.validation import (
    contamination_aware_folds,
    fold_assignment_frame,
    fold_summary,
)


DEFAULT_DATA_DIR = Path("teknofest2026_artificialintelligenceinhealtcare-main")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Part IV, Section D validation strategy."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    prepared = prepare_data(args.data_dir)
    folds = contamination_aware_folds(
        labels=prepared.master["Label"],
        shared_mask=prepared.master_shared_mask,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    assignments = fold_assignment_frame(
        prepared.master,
        prepared.master_shared_mask,
        folds,
    )
    summary = fold_summary(prepared.master, folds)

    assignments.to_csv(args.out_dir / "master_contamination_aware_folds.csv", index=False)
    summary.to_csv(args.out_dir / "master_contamination_aware_fold_summary.csv", index=False)

    panel_unique_summary = {
        "kanser_unique_n": len(prepared.kanser_unique),
        "pah_unique_n": len(prepared.pah_unique),
        "cftr_unique_n": len(prepared.cftr_unique),
    }

    print("Third section validation strategy complete.")
    print(summary.to_string(index=False))
    print(
        "Panel-unique generalization sets: "
        f"KANSER={panel_unique_summary['kanser_unique_n']}, "
        f"PAH={panel_unique_summary['pah_unique_n']}, "
        f"CFTR={panel_unique_summary['cftr_unique_n']}"
    )
    print(f"Fold files written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
