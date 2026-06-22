import numpy as np
import pandas as pd

from medical_metrics import aggregate_fold_metrics, compute_medical_metrics


def test_medical_metrics_include_clinical_confusion_counts_and_scores():
    metrics = compute_medical_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], threshold=0.5)

    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["specificity"] == 0.5
    assert metrics["pathogenic_recall"] == 0.5
    assert metrics["f1"] == metrics["f1_binary"]
    assert 0.0 <= metrics["medical_utility_score"] <= 1.0
    assert 0.0 <= metrics["clinical_safety_score"] <= 1.0


def test_fold_aggregate_has_bootstrap_confidence_interval():
    summary = aggregate_fold_metrics(pd.DataFrame({"fold": [0, 1, 2], "mcc": [0.4, 0.5, 0.6]}), bootstrap_iterations=100)
    row = summary[summary["metric"].eq("mcc")].iloc[0]

    assert row["ci95_low"] <= row["mean"] <= row["ci95_high"]
    assert row["n_folds"] == 3
