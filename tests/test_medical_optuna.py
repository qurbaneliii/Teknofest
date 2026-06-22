import optuna

from teknofest.training import _medical_search_params


def test_medical_optuna_search_space_is_controlled_and_bounded():
    study = optuna.create_study()
    params = _medical_search_params(study.ask(), max_estimators=2000)

    assert 600 <= params["n_estimators"] <= 2000
    assert 0.01 <= params["learning_rate"] <= 0.06
    assert 15 <= params["num_leaves"] <= 127
    assert 3 <= params["max_depth"] <= 8
    assert 30 <= params["min_child_samples"] <= 150
    assert 0.65 <= params["subsample"] <= 0.95
    assert 0.55 <= params["colsample_bytree"] <= 0.95
    assert 0.01 <= params["reg_alpha"] <= 10.0
    assert 0.5 <= params["reg_lambda"] <= 15.0
    assert 0.0 <= params["min_split_gain"] <= 0.1
    assert 0.30 <= params["scale_pos_weight"] <= 0.70
