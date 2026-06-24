# Feature Engineering Audit

`FeatureEngineer` creates missingness, AF, EK, AA, categorical, PCA, and smoothed target-encoding features. In the audited CV run its `.fit()` is invoked only on each fold's training frame, and inference reindexes to the serialized `feature_columns` list. `model_columns` excludes Variant_ID and Label. Remaining limitation: target encodings are valid only when this fold-safe protocol is preserved.
