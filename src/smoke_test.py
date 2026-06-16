from __future__ import annotations

from pathlib import Path

from data_loading import discover_data_dir, write_data_diagnostics
from teknofest.data_prep import prepare_data
from teknofest.validation import contamination_aware_folds


def main() -> None:
    data_dir = discover_data_dir(Path.cwd())
    prepared = prepare_data(data_dir)
    write_data_diagnostics(data_dir)
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    assert folds
    assert all(not prepared.master_shared_mask.iloc[idx] for fold in folds for idx in fold.val_idx)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
