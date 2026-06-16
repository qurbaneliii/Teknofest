from pathlib import Path

from data_loading import discover_data_dir
from teknofest.data_prep import prepare_data
from teknofest.features import FeatureEngineer, detect_binary_al_cols


def test_feature_engineering_alignment_and_variant_id_exclusion_from_model_list():
    prepared = prepare_data(discover_data_dir(Path.cwd()))
    flags = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flags)
    train = engineer.fit_transform(prepared.master.iloc[:100])
    val = engineer.transform(prepared.master.iloc[100:120])
    assert "miss_AL1_6" in val.columns
    assert "cat1_te" in val.columns
    assert set(train.columns) == set(val.columns)

