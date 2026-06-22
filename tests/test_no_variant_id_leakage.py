import pandas as pd

from feature_stability_selection import numeric_model_features
from teknofest.training import model_columns


def test_variant_id_is_never_a_model_feature():
    frame = pd.DataFrame({"Variant_ID": ["V1", "V2"], "Label": [0, 1], "EK_1": [0.1, 0.9]})

    assert "Variant_ID" not in numeric_model_features(frame)
    assert "Variant_ID" not in model_columns(frame)
