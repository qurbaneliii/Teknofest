import numpy as np

from final_ensembling import _cross_fitted_stack, optimize_weights


def test_optimized_ensemble_weights_are_nonnegative_and_normalized():
    matrix = np.array([[0.1, 0.2], [0.2, 0.3], [0.7, 0.6], [0.9, 0.8]])
    weights = optimize_weights(matrix, np.array([0, 0, 1, 1]), objective="medical_utility")

    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_stacking_returns_one_oof_probability_per_row():
    matrix = np.array([[0.1, 0.2], [0.2, 0.3], [0.7, 0.6], [0.9, 0.8], [0.3, 0.2], [0.8, 0.9]])
    probabilities = _cross_fitted_stack(matrix, np.array([0, 0, 1, 1, 0, 1]), np.array([0, 0, 1, 1, 2, 2]), "logistic_stacking")

    assert probabilities.shape == (6,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
