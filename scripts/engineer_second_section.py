from __future__ import annotations

import argparse
from pathlib import Path

from teknofest.data_prep import prepare_data
from teknofest.features import FeatureEngineer, detect_binary_al_cols


DEFAULT_DATA_DIR = Path("teknofest2026_artificialintelligenceinhealtcare-main")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Part IV, Section C feature engineering from the master prompt."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    prepared = prepare_data(args.data_dir)
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(
        al_cols=prepared.al_cols,
        al_raw=prepared.al_raw,
        flag_cols=flag_cols,
    )

    outputs = {
        "master_engineered": engineer.fit_transform(prepared.master),
        "kanser_engineered": engineer.transform(prepared.kanser),
        "pah_engineered": engineer.transform(prepared.pah),
        "cftr_engineered": engineer.transform(prepared.cftr),
        "kanser_unique_engineered": engineer.transform(prepared.kanser_unique),
        "pah_unique_engineered": engineer.transform(prepared.pah_unique),
        "cftr_unique_engineered": engineer.transform(prepared.cftr_unique),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(args.out_dir / f"{name}.csv", index=False)

    feature_cols = sorted(
        set(outputs["master_engineered"].columns) - set(prepared.master.columns)
    )
    (args.out_dir / "engineered_feature_columns.txt").write_text(
        "\n".join(feature_cols) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "binary_al_flag_columns.txt").write_text(
        "\n".join(flag_cols) + "\n",
        encoding="utf-8",
    )

    print("Second section feature engineering complete.")
    print(f"Binary AL flag columns detected: {len(flag_cols)}")
    print(f"New engineered feature columns: {len(feature_cols)}")
    print(f"MASTER engineered shape: {outputs['master_engineered'].shape}")
    print(f"Processed files written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
