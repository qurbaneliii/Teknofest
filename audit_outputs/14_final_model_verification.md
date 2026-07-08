# Final Model Verification

Final audited candidate: `audited_master_only_lgbm`.

| dataset_split         |    n |   threshold |   roc_auc |   pr_auc |       f1 |   f1_macro |      mcc |   precision |   pathogenic_recall |   tn |   fp |   fn |   tp |
|:----------------------|-----:|------------:|----------:|---------:|---------:|-----------:|---------:|------------:|--------------------:|-----:|-----:|-----:|-----:|
| MASTER_ONLY_CV        | 2353 |     0.43815 |  0.847814 | 0.904085 | 0.866156 |   0.781447 | 0.563587 |    0.854897 |            0.877716 |  502 |  240 |  197 | 1414 |
| CFTR_UNIQUE           |   34 |     0.43815 |  0.961938 | 0.961641 | 0.882353 |   0.882353 | 0.764706 |    0.882353 |            0.882353 |   15 |    2 |    2 |   15 |
| KANSER_UNIQUE         |  142 |     0.43815 |  0.906071 | 0.800442 | 0.743363 |   0.786886 | 0.619632 |    0.617647 |            0.933333 |   71 |   26 |    3 |   42 |
| PAH_UNIQUE            |  117 |     0.43815 |  0.821128 | 0.854474 | 0.813793 |   0.755211 | 0.520365 |    0.766234 |            0.867647 |   31 |   18 |    9 |   59 |
| PANEL_UNIQUE_COMBINED |  293 |     0.43815 |  0.876168 | 0.825512 | 0.794521 |   0.795219 | 0.609639 |    0.716049 |            0.892308 |  117 |   46 |   14 |  116 |

No official unlabeled test file was found. Hidden-test metrics are unavailable and are not estimated here.
