import pandas as pd

from teknofest.features import FeatureEngineer


def _raw_frame(labels, categories):
    frame = pd.DataFrame(
        {
            "Variant_ID": [f"V{index}" for index in range(len(labels))],
            "Label": labels,
            "CAT_1": categories,
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


def test_target_encoding_is_fit_only_from_training_fold_labels():
    train = _raw_frame([0, 1, 1], ["train_a", "train_a", "train_b"])
    validation = _raw_frame([0], ["validation_only"])
    engineer = FeatureEngineer(
        al_cols=[f"AL_{index}" for index in range(1, 39)],
        al_raw=[f"AL_{index}" for index in range(1, 27)],
        flag_cols=[],
    ).fit(train)
    transformed = engineer.transform(validation)

    assert transformed.loc[0, "cat1_te"] == train["Label"].mean()
