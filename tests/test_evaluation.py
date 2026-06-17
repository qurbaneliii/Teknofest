import pandas as pd

from evaluation import diagnose_model_performance


def test_diagnose_model_performance_detects_thresholding_issue():
    metrics = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "evaluation_split": "MASTER_CV",
                "threshold_type": "default_0.5",
                "roc_auc": 0.84,
                "f1_macro": 0.48,
                "mcc": 0.20,
            },
            {
                "model_name": "lightgbm",
                "evaluation_split": "MASTER_CV",
                "threshold_type": "f1_macro_opt",
                "roc_auc": 0.84,
                "f1_macro": 0.75,
                "mcc": 0.52,
            },
            {
                "model_name": "lightgbm",
                "evaluation_split": "panel_unique_combined",
                "threshold_type": "saved_panel_threshold",
                "roc_auc": 0.86,
                "f1_macro": 0.75,
                "mcc": 0.56,
            },
        ]
    )

    diagnosis = diagnose_model_performance(metrics)

    assert diagnosis["model_strength"] == "moderate"
    assert diagnosis["main_issue"] == "thresholding"
