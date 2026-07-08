# Final Contract Status

```json
{
  "final_contract_complete": true,
  "final_model_id": "lightgbm_conservative_regularized",
  "final_threshold": 0.471,
  "calibration": "none",
  "ensemble_replaced_lightgbm": false,
  "protected_baseline_overwritten": false,
  "official_metric_claimed": false,
  "hgb_promoted": false,
  "missing_files": [],
  "failed_checks": [],
  "warnings": [
    "No official test CSV was found locally."
  ],
  "final_decision": "protected LightGBM baseline locked as final",
  "next_action": "Use the locked inference CLI only when organizer-format test data is supplied.",
  "pytest_passed": true
}
```

Pytest output:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\Users\qurba\OneDrive\Documents\Teknofest
plugins: anyio-4.13.0, langsmith-0.7.37
collected 18 items

tests\test_advanced_bio_features.py .                                    [  5%]
tests\test_data_loading.py ..                                            [ 16%]
tests\test_ensembling.py ..                                              [ 27%]
tests\test_evaluation.py .                                               [ 33%]
tests\test_features.py .                                                 [ 38%]
tests\test_final_inference_schema.py .                                   [ 44%]
tests\test_final_model_zoo.py .                                          [ 50%]
tests\test_final_selection_board.py .                                    [ 55%]
tests\test_fold_safe_target_encoding.py .                                [ 61%]
tests\test_medical_metrics.py ..                                         [ 72%]
tests\test_medical_optuna.py .                                           [ 77%]
tests\test_no_variant_id_leakage.py .                                    [ 83%]
tests\test_thresholding.py .                                             [ 88%]
tests\test_validation.py ..                                              [100%]

============================== warnings summary ===============================
tests/test_final_inference_schema.py::test_final_inference_accepts_unlabeled_organizer_format_rows
tests/test_fold_safe_target_encoding.py::test_target_encoding_is_fit_only_from_training_fold_labels
  C:\Users\qurba\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\decomposition\_pca.py:646: RuntimeWarning: invalid value encountered in divide
    explained_variance_ratio_ = explained_variance_ / total_var

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 18 passed, 2 warnings in 23.24s =======================

```
