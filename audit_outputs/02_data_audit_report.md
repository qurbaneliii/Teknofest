# Data Audit

| dataset   |   rows |   columns | target_column   |   pathogenic |   benign |   pathogenic_rate |
|:----------|-------:|----------:|:----------------|-------------:|---------:|------------------:|
| MASTER    |   2931 |       353 | Label           |         2149 |      782 |          0.733197 |
| KANSER    |    388 |       353 | Label           |          268 |      120 |          0.690722 |
| PAH       |    372 |       353 | Label           |          310 |       62 |          0.833333 |
| CFTR      |    111 |       353 | Label           |           90 |       21 |          0.810811 |

## Schema

| dataset   | schema_matches_master   | missing_columns   | extra_columns   | order_matches_master   |
|:----------|:------------------------|:------------------|:----------------|:-----------------------|
| MASTER    | True                    |                   |                 | True                   |
| KANSER    | True                    |                   |                 | True                   |
| PAH       | True                    |                   |                 | True                   |
| CFTR      | True                    |                   |                 | True                   |

## Warnings

- No unexpected target-like names found beyond Label.

Duplicate and overlap details are in `duplicate_overlap_summary.csv`; missingness details are in `missingness_summary.csv`.
