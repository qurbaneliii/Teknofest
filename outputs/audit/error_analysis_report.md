# Error Analysis

False negatives are clinically more concerning because pathogenic variants may be missed. KANSER has high recall but comparatively lower precision; PAH has the weakest MCC among panels. CFTR has only 34 unique rows and must not be over-interpreted.

| evaluation_split   | error_type     |    n |
|:-------------------|:---------------|-----:|
| CFTR_UNIQUE        | correct        |   30 |
| CFTR_UNIQUE        | false_negative |    2 |
| CFTR_UNIQUE        | false_positive |    2 |
| KANSER_UNIQUE      | correct        |  113 |
| KANSER_UNIQUE      | false_negative |    3 |
| KANSER_UNIQUE      | false_positive |   26 |
| MASTER_ONLY_CV     | correct        | 1916 |
| MASTER_ONLY_CV     | false_negative |  197 |
| MASTER_ONLY_CV     | false_positive |  240 |
| PAH_UNIQUE         | correct        |   90 |
| PAH_UNIQUE         | false_negative |    9 |
| PAH_UNIQUE         | false_positive |   18 |

Error rows and confidence bands are in `outputs/final/error_analysis.csv`.
