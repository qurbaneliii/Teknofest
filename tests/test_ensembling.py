import numpy as np

from final_ensembling import optimize_weights


def test_optimized_ensemble_weights_are_nonnegative_and_normalized():
    matrix = np.array([[0.1, 0.2], [0.2, 0.3], [0.7, 0.6], [0.9, 0.8]])
    weights = optimize_weights(matrix, np.array([0, 0, 1, 1]), target="medical_utility")

    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
