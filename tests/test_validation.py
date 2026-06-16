from pathlib import Path

import numpy as np

from data_loading import discover_data_dir
from teknofest.data_prep import prepare_data
from teknofest.validation import contamination_aware_folds, best_f1_macro_threshold


def test_validation_folds_exclude_master_shared_variants():
    prepared = prepare_data(discover_data_dir(Path.cwd()))
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    assert all(not prepared.master_shared_mask.iloc[idx] for fold in folds for idx in fold.val_idx)


def test_threshold_helper_on_synthetic_arrays():
    threshold, score = best_f1_macro_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert 0.0 <= threshold <= 1.0
    assert score == 1.0
