import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from final_inference import generate_final_submission
from teknofest.features import FeatureEngineer
from teknofest.training import model_columns


def _raw_frame(labels):
    frame = pd.DataFrame(
        {
            "Variant_ID": [f"V{index}" for index in range(len(labels))],
            "Label": labels,
            "CAT_1": ["a"] * len(labels),
            "CAT_2": ["x"] * len(labels),
            "CAT_3": ["0/1"] * len(labels),
            "CAT_4": ["0/1"] * len(labels),
            "CAT_5": ["0/1"] * len(labels),
            "CAT_6": [None] * len(labels),
            "AA_1": ["A"] * len(labels),
            "AA_2": ["V"] * len(labels),
        }
    )
    for index in range(1, 39):
        frame[f"AL_{index}"] = 0.0
    for index in range(1, 10):
        frame[f"EK_{index}"] = float(index)
    return frame


def test_final_inference_accepts_unlabeled_organizer_format_rows(tmp_path):
    train = _raw_frame([0, 1, 1, 0])
    engineer = FeatureEngineer([f"AL_{index}" for index in range(1, 39)], [f"AL_{index}" for index in range(1, 27)], []).fit(train)
    engineered = engineer.transform(train)
    columns = model_columns(engineered)
    model = DummyClassifier(strategy="prior").fit(engineered[columns], train["Label"])
    bundle_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "feature_engineer": engineer, "advanced_feature_engineer": None, "feature_columns": columns}, bundle_path)
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps({"model_id": "test_model", "model_kind": "single_model", "artifact_path": str(bundle_path), "threshold": 0.5, "calibration": "none"}),
        encoding="utf-8",
    )

    unlabeled = train.drop(columns="Label")
    output, destination = generate_final_submission(unlabeled, tmp_path / "submission.csv", decision_path)

    assert destination.exists()
    assert {"Variant_ID", "predicted_probability", "predicted_label", "threshold_used", "model_id", "uncertainty_flag"}.issubset(output.columns)
