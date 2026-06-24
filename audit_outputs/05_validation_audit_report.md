# Validation Audit

The audited design is 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on MASTER-only variants. Every MASTER ID appearing in a supplied panel is excluded from both train and validation. Feature PCA, categorical dummy schema, and target encodings are fit within each fold.

|   fold |   n |   pathogenic_recall |   specificity |   f1_macro |      mcc |   fold_selected_threshold |
|-------:|----:|--------------------:|--------------:|-----------:|---------:|--------------------------:|
|      0 | 471 |            0.86646  |      0.597315 |   0.738814 | 0.480222 |                  0.455224 |
|      1 | 471 |            0.878882 |      0.704698 |   0.794417 | 0.589069 |                  0.467607 |
|      2 | 471 |            0.869969 |      0.743243 |   0.804361 | 0.608872 |                  0.358997 |
|      3 | 470 |            0.881988 |      0.689189 |   0.789749 | 0.580107 |                  0.43815  |
|      4 | 470 |            0.891304 |      0.648649 |   0.778277 | 0.559344 |                  0.288836 |
