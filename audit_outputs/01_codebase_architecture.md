# Codebase Architecture

Raw organizer CSVs are loaded by `teknofest.data_prep`; the legacy pipeline applies `FeatureEngineer`, trains LightGBM-family models, writes OOF/panel predictions, then derives reports. `src/final_inference.py` loads a serialized bundle and a decision JSON. The legacy `run_pipeline.py` and final-selection scripts coexist with a V3 pipeline and substantial archived artifacts, so a filename alone is not provenance.

This audit uses only the organizer training files, excludes every MASTER variant shared with KANSER/PAH/CFTR before both training and validation, fits feature engineering independently within each fold, serializes a new bundle, and writes fresh predictions under `artifacts/predictions/audited_*`.
