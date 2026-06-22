import numpy as np

from final_thresholding import select_threshold_candidates, threshold_grid


def test_threshold_candidates_cover_medical_and_safety_policies():
    grid = threshold_grid(np.array([0, 0, 1, 1]), np.array([0.05, 0.45, 0.55, 0.95]))
    candidates = select_threshold_candidates(grid)

    assert {"max_medical_utility", "max_clinical_safety", "recall_ge_0_90_best_specificity", "youden_j"}.issubset(candidates["threshold_strategy"])
    assert candidates["threshold"].dropna().between(0.01, 0.99).all()
