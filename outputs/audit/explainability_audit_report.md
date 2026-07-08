# Explainability Audit

The importance file is generated directly from the audited serialized LightGBM artifact. Variant_ID and Label are not model features. Importance is predictive split importance, not biological causality. Legacy SHAP files do not represent this audited model.

| feature_group             |   importance |
|:--------------------------|-------------:|
| AL / population frequency |         5009 |
| EK / computational        |         2174 |
| missingness               |          448 |
| interaction               |          434 |
| AA / amino-acid           |          375 |
| CAT / metadata            |          109 |
| other                     |           79 |
