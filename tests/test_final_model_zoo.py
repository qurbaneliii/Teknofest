from final_model_zoo import default_model_specs


def test_phase2_model_zoo_has_only_requested_model_families():
    specs = default_model_specs()

    assert [spec.model_id for spec in specs] == [
        "lightgbm_conservative_regularized",
        "lightgbm_high_capacity_controlled",
        "catboost",
        "xgboost",
        "extra_trees",
        "elasticnet_logistic_regression",
    ]
