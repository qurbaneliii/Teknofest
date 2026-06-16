from pathlib import Path

from data_loading import discover_data_dir, validate_schema
from teknofest.data_prep import load_datasets, prepare_data


def test_data_files_load_and_schema_is_valid():
    data_dir = discover_data_dir(Path.cwd())
    datasets = load_datasets(data_dir)
    validate_schema(datasets)
    assert {"master", "kanser", "pah", "cftr"} == set(datasets)


def test_al_185_is_dropped_after_preparation():
    prepared = prepare_data(discover_data_dir(Path.cwd()))
    assert "AL_185" not in prepared.master.columns
    assert "AL_186" in prepared.master.columns

