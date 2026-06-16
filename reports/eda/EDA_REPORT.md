# TEKNOFEST 2026 EDA Report

## Executive Summary

The 10 most predictive individual numeric features on MASTER by Mann-Whitney AUC-equivalent are:

| feature      |   auc_equivalent |   auc_directional_strength |     p_value |   n_notna |
|:-------------|-----------------:|---------------------------:|------------:|----------:|
| EK_7         |         0.738042 |                   0.738042 | 5.72001e-73 |      2572 |
| EK7xEK9      |         0.726059 |                   0.726059 | 3.57778e-60 |      2324 |
| EK_9         |         0.695513 |                   0.695513 | 1.77982e-45 |      2324 |
| EK_3         |         0.690208 |                   0.690208 | 3.4529e-33  |      1601 |
| n_pops       |         0.316095 |                   0.683905 | 2.95749e-57 |      2931 |
| EK_2         |         0.67377  |                   0.67377  | 1.0122e-39  |      2572 |
| n_nonmiss_AL |         0.336843 |                   0.663157 | 2.73369e-42 |      2931 |
| AL_287       |         0.342764 |                   0.657236 | 5.30989e-17 |       979 |
| AL_215       |         0.35167  |                   0.64833  | 2.05825e-12 |       752 |
| AL_26        |         0.355282 |                   0.644718 | 9.56751e-24 |      1846 |

The single most important AL structural discovery is that AL columns cluster strongly by identical non-null counts. The largest discovered availability groups are:

|   n_notna |   columns |
|----------:|----------:|
|      1041 |        95 |
|      1486 |        84 |
|      1632 |        40 |
|      1514 |        36 |
|      1197 |        24 |
|       252 |        12 |
|      1787 |         9 |
|       294 |         6 |
|      1055 |         5 |
|      1056 |         5 |

The most important missingness finding is that missingness itself is label-associated for many columns. The strongest missingness signals are:

| column   |   missing_rate |   label_rate_when_missing |   label_rate_when_present |   missing_label_corr |
|:---------|---------------:|--------------------------:|--------------------------:|---------------------:|
| AL_4     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_3     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_1     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_2     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_6     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_5     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_29    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_30    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_32    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_37    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |

The most important cross-dataset finding is that panel-unique variants have very different pathogenic rates from panel variants shared with MASTER:

| panel   | subset             |   n |   pathogenic_rate |   overall_missing_rate |   numeric_range_mean |
|:--------|:-------------------|----:|------------------:|-----------------------:|---------------------:|
| KANSER  | unique             | 142 |          0.316901 |               0.427742 |             0.647921 |
| KANSER  | shared_with_master | 246 |          0.906504 |               0.654875 |             0.828301 |
| PAH     | unique             | 117 |          0.581197 |               0.576814 |             0.487153 |
| PAH     | shared_with_master | 255 |          0.94902  |               0.527445 |             0.531461 |
| CFTR    | unique             |  34 |          0.5      |               0.224546 |             0.552333 |
| CFTR    | shared_with_master |  77 |          0.948052 |               0.337331 |             0.554045 |

Leakage suspects with AUC-equivalent directional strength above 0.85:

None above 0.85.

## Phase 1 - Data Integrity And Structure

All four files were loaded from the raw competition folder and compared against MASTER column order.

| dataset   |   rows |   columns | schema_identical_to_master   |   duplicate_variant_id |   pathogenic_rate |   pathogenic |   benign |
|:----------|-------:|----------:|:-----------------------------|-----------------------:|------------------:|-------------:|---------:|
| MASTER    |   2931 |       353 | True                         |                      0 |          0.733197 |         2149 |      782 |
| KANSER    |    388 |       353 | True                         |                      0 |          0.690722 |          268 |      120 |
| PAH       |    372 |       353 | True                         |                      0 |          0.833333 |          310 |       62 |
| CFTR      |    111 |       353 | True                         |                      0 |          0.810811 |           90 |       21 |

Overlap counts and panel-unique subsets:

| panel   |   master_overlap |   panel_unique |   panel_unique_pathogenic_rate |   panel_shared_pathogenic_rate |
|:--------|-----------------:|---------------:|-------------------------------:|-------------------------------:|
| KANSER  |              246 |            142 |                       0.316901 |                       0.906504 |
| PAH     |              255 |            117 |                       0.581197 |                       0.94902  |
| CFTR    |               77 |             34 |                       0.5      |                       0.948052 |

MASTER-only has n=2353 and pathogenic rate 0.6847. MASTER-shared-with-panel has n=578 and pathogenic rate 0.9308.

Constant or near-constant columns (>99% same non-null value):

| column   |   n_nonnull |   n_unique_nonnull |   top_value |   top_value_rate_nonnull |
|:---------|------------:|-------------------:|------------:|-------------------------:|
| AL_80    |        1197 |                  1 |           1 |                        1 |
| AL_101   |        1486 |                  1 |           1 |                        1 |
| AL_104   |        1486 |                  1 |           1 |                        1 |
| AL_107   |        1486 |                  1 |           1 |                        1 |
| AL_110   |        1486 |                  1 |           1 |                        1 |
| AL_113   |        1486 |                  1 |           1 |                        1 |
| AL_116   |        1486 |                  1 |           1 |                        1 |
| AL_119   |        1486 |                  1 |           1 |                        1 |
| AL_122   |        1486 |                  1 |           1 |                        1 |
| AL_125   |        1486 |                  1 |           1 |                        1 |
| AL_128   |        1486 |                  1 |           1 |                        1 |
| AL_131   |        1486 |                  1 |           1 |                        1 |
| AL_134   |        1486 |                  1 |           1 |                        1 |
| AL_137   |        1486 |                  1 |           1 |                        1 |
| AL_140   |        1486 |                  1 |           1 |                        1 |
| AL_143   |        1486 |                  1 |           1 |                        1 |
| AL_146   |        1486 |                  1 |           1 |                        1 |
| AL_149   |        1486 |                  1 |           1 |                        1 |
| AL_152   |        1486 |                  1 |           1 |                        1 |
| AL_155   |        1486 |                  1 |           1 |                        1 |
| AL_158   |        1486 |                  1 |           1 |                        1 |
| AL_161   |        1486 |                  1 |           1 |                        1 |
| AL_164   |        1486 |                  1 |           1 |                        1 |
| AL_167   |        1486 |                  1 |           1 |                        1 |
| AL_170   |        1486 |                  1 |           1 |                        1 |

Missingness bands:

| band   |   columns |
|:-------|----------:|
| 26-50% |       173 |
| 51-75% |       146 |
| 76-99% |        19 |
| 1-25%  |        13 |
| 0%     |         2 |

Figure: `reports\eda\figures\missingness_top80.png`

## Phase 2 - Univariate Analysis By Feature Group

### AL Columns

AL columns were profiled for distribution, missingness, binary status, and grouped by identical non-null counts. There are 103 strictly binary AL columns in MASTER.

Top AL availability groups:

|   n_notna |   columns |
|----------:|----------:|
|      1041 |        95 |
|      1486 |        84 |
|      1632 |        40 |
|      1514 |        36 |
|      1197 |        24 |
|       252 |        12 |
|      1787 |         9 |
|       294 |         6 |
|      1055 |         5 |
|      1056 |         5 |

### CAT Columns

| column   |   unique_values |   missing_rate |   composite_rate |
|:---------|----------------:|---------------:|-----------------:|
| CAT_1    |              30 |       0.370181 |      0.020812    |
| CAT_2    |               7 |       0.591607 |      0           |
| CAT_3    |               5 |       0.122484 |      0           |
| CAT_4    |               5 |       0.122484 |      0           |
| CAT_5    |               5 |       0.122484 |      0           |
| CAT_6    |               3 |       0.977141 |      0.000682361 |

Most label-skewed CAT categories:

| column   | value                                                                                                                         |   count |   label_rate |   delta_from_base_rate |
|:---------|:------------------------------------------------------------------------------------------------------------------------------|--------:|-------------:|-----------------------:|
| CAT_1    | gnomADe_NFE&gnomADg_NFE                                                                                                       |       1 |     0        |              -0.733197 |
| CAT_6    | lcr                                                                                                                           |       2 |     0        |              -0.733197 |
| CAT_1    | AFR                                                                                                                           |      61 |     0.180328 |              -0.552869 |
| CAT_1    | SAS                                                                                                                           |      32 |     0.28125  |              -0.451947 |
| CAT_1    | EAS                                                                                                                           |      20 |     0.3      |              -0.433197 |
| CAT_1    | gnomADg_MID                                                                                                                   |      20 |     0.3      |              -0.433197 |
| CAT_2    | AllofUs_SAS                                                                                                                   |      81 |     0.432099 |              -0.301098 |
| CAT_2    | AllofUs_MID                                                                                                                   |      40 |     0.45     |              -0.283197 |
| CAT_1    | gnomADg_AMI                                                                                                                   |      13 |     0.461538 |              -0.271658 |
| CAT_1    | EAS&EUR                                                                                                                       |       1 |     1        |               0.266803 |
| CAT_1    | EAS&SAS                                                                                                                       |       2 |     1        |               0.266803 |
| CAT_1    | gnomADg_AFR&gnomADg_AMI&gnomADg_AMR&gnomADg_ASJ&gnomADg_EAS&gnomADg_FIN&gnomADg_MID&gnomADg_NFE&gnomADg_REMAINING&gnomADg_SAS |       5 |     1        |               0.266803 |
| CAT_1    | EAS&EUR&SAS&gnomADe_ASJ&gnomADe_FIN&gnomADg_AMI&gnomADg_ASJ&gnomADg_EAS&gnomADg_FIN&gnomADg_MID&gnomADg_SAS                   |       1 |     1        |               0.266803 |
| CAT_6    | segdup                                                                                                                        |      63 |     0.47619  |              -0.257006 |
| CAT_1    | EUR                                                                                                                           |      35 |     0.485714 |              -0.247483 |
| CAT_1    | gnomADe_FIN                                                                                                                   |      34 |     0.5      |              -0.233197 |
| CAT_6    | decoy&segdup                                                                                                                  |       2 |     0.5      |              -0.233197 |
| CAT_1    | gnomADe_AFR&gnomADe_AMR&gnomADe_ASJ&gnomADe_EAS&gnomADe_FIN&gnomADe_MID&gnomADe_NFE&gnomADe_REMAINING&gnomADe_SAS             |      51 |     0.960784 |               0.227587 |
| CAT_2    | AllofUs_AFR                                                                                                                   |     236 |     0.529661 |              -0.203536 |
| CAT_1    | gnomADg_REMAINING                                                                                                             |      39 |     0.564103 |              -0.169094 |

### EK Columns

| feature   |   count |     mean |      std |        min |      25% |      50% |       75% |      max |   missing_rate | has_negative   |
|:----------|--------:|---------:|---------:|-----------:|---------:|---------:|----------:|---------:|---------------:|:---------------|
| EK_1      |    2572 | 5.26338  | 0.594258 |   0.855571 | 4.97636  | 5.36523  |  5.67606  |  6.17    |       0.122484 | False          |
| EK_2      |    2572 | 4.46425  | 2.12725  | -11.9      | 4.29394  | 5.09838  |  5.59092  |  6.17    |       0.122484 | True           |
| EK_3      |    1601 | 3.0462   | 2.5052   | -11.3689   | 2.60333  | 3.69173  |  4.44145  |  7.01042 |       0.45377  | True           |
| EK_4      |    2572 | 0.896647 | 0.265615 |   0        | 0.962463 | 0.991925 |  1        |  1       |       0.122484 | False          |
| EK_5      |    2572 | 0.805125 | 0.305456 |   0        | 0.770834 | 0.960908 |  0.997708 |  1       |       0.122484 | False          |
| EK_6      |    2572 | 0.905764 | 0.259208 |   0        | 0.967497 | 0.9945   |  1        |  1       |       0.122484 | False          |
| EK_7      |    2572 | 5.94423  | 3.17259  |  -3.32195  | 3.43363  | 7.10006  |  8.41847  | 10.003   |       0.122484 | True           |
| EK_8      |    2572 | 0.559148 | 0.34474  |  -5.716    | 0.579527 | 0.639698 |  0.692865 |  0.756   |       0.122484 | True           |
| EK_9      |    2324 | 7.48213  | 3.90584  | -20        | 5.81013  | 7.85384  | 11.0399   | 11.934   |       0.207097 | True           |

EK feature pairs with |Spearman rho| > 0.5:

| feature_1   | feature_2   |   spearman_rho |
|:------------|:------------|---------------:|
| EK_7        | EK_9        |       0.753888 |
| EK_3        | EK_7        |       0.638417 |
| EK_1        | EK_2        |       0.583466 |
| EK_3        | EK_9        |       0.569274 |
| EK_2        | EK_3        |       0.567391 |

Figure: `reports\eda\figures\ek_spearman_correlation.png`

### AA Columns

Most common amino-acid substitutions:

| aa_pair   |   count |
|:----------|--------:|
| G>R       |      84 |
| R>C       |      80 |
| L>P       |      75 |
| E>K       |      75 |
| R>H       |      74 |
| R>Q       |      73 |
| P>L       |      71 |
| R>W       |      68 |
| C>Y       |      54 |
| D>N       |      47 |
| Y>C       |      45 |
| C>R       |      43 |
| A>V       |      40 |
| A>T       |      39 |
| P>S       |      39 |
| G>S       |      37 |
| G>D       |      37 |
| G>E       |      33 |
| V>M       |      32 |
| D>G       |      31 |

## Phase 3 - Bivariate Analysis: Features vs Label

Top 25 numeric features:

| feature      |   auc_equivalent |   auc_directional_strength |     p_value |   rank_biserial |   n_notna |
|:-------------|-----------------:|---------------------------:|------------:|----------------:|----------:|
| EK_7         |         0.738042 |                   0.738042 | 5.72001e-73 |        0.476084 |      2572 |
| EK7xEK9      |         0.726059 |                   0.726059 | 3.57778e-60 |        0.452118 |      2324 |
| EK_9         |         0.695513 |                   0.695513 | 1.77982e-45 |        0.391025 |      2324 |
| EK_3         |         0.690208 |                   0.690208 | 3.4529e-33  |        0.380417 |      1601 |
| n_pops       |         0.316095 |                   0.683905 | 2.95749e-57 |       -0.36781  |      2931 |
| EK_2         |         0.67377  |                   0.67377  | 1.0122e-39  |        0.347539 |      2572 |
| n_nonmiss_AL |         0.336843 |                   0.663157 | 2.73369e-42 |       -0.326314 |      2931 |
| AL_287       |         0.342764 |                   0.657236 | 5.30989e-17 |       -0.314472 |       979 |
| AL_215       |         0.35167  |                   0.64833  | 2.05825e-12 |       -0.29666  |       752 |
| AL_26        |         0.355282 |                   0.644718 | 9.56751e-24 |       -0.289436 |      1846 |
| EK_4         |         0.637392 |                   0.637392 | 2.30078e-27 |        0.274784 |      2572 |
| AL_186       |         0.366538 |                   0.633462 | 3.61968e-21 |       -0.266923 |      2012 |
| AL_251       |         0.36908  |                   0.63092  | 2.219e-12   |       -0.261839 |       991 |
| AL_73        |         0.369644 |                   0.630356 | 1.39545e-15 |       -0.260713 |      1514 |
| max_AF       |         0.372088 |                   0.627912 | 1.31566e-27 |       -0.255825 |      2931 |
| AL_94        |         0.373917 |                   0.626083 | 3.63229e-13 |       -0.252167 |      1197 |
| AL_79        |         0.374037 |                   0.625963 | 3.82393e-13 |       -0.251926 |      1197 |
| AL_184       |         0.379006 |                   0.620994 | 1.06967e-10 |       -0.241987 |       985 |
| AL_327       |         0.38091  |                   0.61909  | 1.97777e-15 |       -0.23818  |      1632 |
| AL_136       |         0.386709 |                   0.613291 | 1.01842e-12 |       -0.226583 |      1486 |
| AL_224       |         0.390054 |                   0.609946 | 1.78656e-09 |       -0.219892 |      1041 |
| EK_6         |         0.60949  |                   0.60949  | 4.35918e-18 |        0.21898  |      2572 |
| AL_91        |         0.393959 |                   0.606041 | 9.7678e-10  |       -0.212082 |      1197 |
| AL_260       |         0.394473 |                   0.605527 | 7.7277e-09  |       -0.211054 |      1041 |
| AL_178       |         0.398848 |                   0.601152 | 1.96312e-10 |       -0.202304 |      1486 |

Bottom 25 numeric features:

| feature   |   auc_equivalent |   auc_directional_strength |   p_value |   rank_biserial |   n_notna |
|:----------|-----------------:|---------------------------:|----------:|----------------:|----------:|
| AL_312    |              0.5 |                        0.5 |         1 |               0 |      1632 |
| AL_309    |              0.5 |                        0.5 |         1 |               0 |      1632 |
| AL_284    |              0.5 |                        0.5 |         1 |               0 |      1041 |
| AL_272    |              0.5 |                        0.5 |         1 |               0 |      1041 |
| AL_276    |              0.5 |                        0.5 |         1 |               0 |      1041 |
| AL_280    |              0.5 |                        0.5 |         1 |               0 |      1041 |
| AL_307    |              0.5 |                        0.5 |         1 |               0 |      1632 |
| AL_122    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_125    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_131    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_128    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_134    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_101    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_292    |              0.5 |                        0.5 |         1 |               0 |      1041 |
| AL_316    |              0.5 |                        0.5 |         1 |               0 |      1632 |
| AL_170    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_173    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_164    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_161    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_167    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_158    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_155    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_80     |              0.5 |                        0.5 |         1 |               0 |      1197 |
| AL_137    |              0.5 |                        0.5 |         1 |               0 |      1486 |
| AL_140    |              0.5 |                        0.5 |         1 |               0 |      1486 |

Missingness-as-feature ranking:

| column   |   missing_rate |   label_rate_when_missing |   label_rate_when_present |   missing_label_corr |
|:---------|---------------:|--------------------------:|--------------------------:|---------------------:|
| AL_4     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_3     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_1     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_2     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_6     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_5     |       0.899693 |                  0.784224 |                  0.27551  |             0.345526 |
| AL_29    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_30    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_32    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_37    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_33    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_34    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_31    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_28    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_27    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_36    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_35    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_38    |       0.914023 |                  0.775289 |                  0.285714 |             0.310302 |
| AL_19    |       0.639713 |                  0.834667 |                  0.55303  |             0.305703 |
| AL_23    |       0.639713 |                  0.834667 |                  0.55303  |             0.305703 |
| AL_17    |       0.639713 |                  0.834667 |                  0.55303  |             0.305703 |
| AL_16    |       0.639713 |                  0.834667 |                  0.55303  |             0.305703 |
| AL_18    |       0.639713 |                  0.834667 |                  0.55303  |             0.305703 |
| AL_25    |       0.640055 |                  0.834222 |                  0.553555 |             0.304588 |
| AL_20    |       0.640055 |                  0.834222 |                  0.553555 |             0.304588 |

Figure: `reports\eda\figures\top10_numeric_boxplots.png`

## Phase 4 - Multivariate And Structural Analysis

Adjacent AL triplet Spearman correlations show whether apparent triplets are redundant or independent. Lowest mean absolute triplet correlations:

| columns              |       rho_12 |        rho_13 |       rho_23 |   mean_abs_rho |
|:---------------------|-------------:|--------------:|-------------:|---------------:|
| AL_246,AL_247,AL_248 |  -0.0046159  | nan           | nan          |     0.0046159  |
| AL_114,AL_115,AL_116 |  -0.00517469 | nan           | nan          |     0.00517469 |
| AL_255,AL_256,AL_257 | nan          |  -0.00575998  | nan          |     0.00575998 |
| AL_48,AL_49,AL_50    |   0.0122099  |  -0.000520461 |   0.00561069 |     0.00611367 |
| AL_105,AL_106,AL_107 |   0.00737394 | nan           | nan          |     0.00737394 |
| AL_276,AL_277,AL_278 | nan          | nan           |  -0.00769246 |     0.00769246 |
| AL_264,AL_265,AL_266 |  -0.021498   |  -0.00209795  |  -0.00168402 |     0.00842666 |
| AL_117,AL_118,AL_119 |   0.0101094  | nan           | nan          |     0.0101094  |
| AL_138,AL_139,AL_140 |   0.0107197  | nan           | nan          |     0.0107197  |
| AL_51,AL_52,AL_53    |   0.0115731  |   0.000645921 |   0.0218618  |     0.0113603  |
| AL_132,AL_133,AL_134 |  -0.0129731  | nan           | nan          |     0.0129731  |
| AL_102,AL_103,AL_104 |   0.0148507  | nan           | nan          |     0.0148507  |
| AL_300,AL_301,AL_302 |   0.0118066  |   0.00645438  |  -0.0264155  |     0.0148922  |
| AL_72,AL_73,AL_74    |   0.00237824 |   0.0142695   |   0.0292899  |     0.0153126  |
| AL_315,AL_316,AL_317 | nan          |  -0.0157669   | nan          |     0.0157669  |

Highest mean absolute triplet correlations:

| columns              |     rho_12 |       rho_13 |      rho_23 |   mean_abs_rho |
|:---------------------|-----------:|-------------:|------------:|---------------:|
| AL_267,AL_268,AL_269 | -0.0297951 |   1          |  -0.0297951 |      0.353197  |
| AL_27,AL_28,AL_29    |  0.268911  |   0.0658111  |   0.248208  |      0.19431   |
| AL_36,AL_37,AL_38    |  0.120672  |   0.18469    |   0.212295  |      0.172553  |
| AL_297,AL_298,AL_299 |  0.121246  | nan          | nan         |      0.121246  |
| AL_186,AL_187,AL_188 |  0.0858564 |   0.228541   |   0.0348032 |      0.1164    |
| AL_30,AL_31,AL_32    |  0.137067  |   0.0674288  |   0.141929  |      0.115475  |
| AL_33,AL_34,AL_35    |  0.152766  |   0.0443964  |   0.101536  |      0.099566  |
| AL_90,AL_91,AL_92    |  0.0764224 |   0.0526495  |   0.15369   |      0.0942541 |
| AL_120,AL_121,AL_122 |  0.0941086 | nan          | nan         |      0.0941086 |
| AL_171,AL_172,AL_173 |  0.091057  | nan          | nan         |      0.091057  |
| AL_129,AL_130,AL_131 |  0.0814679 | nan          | nan         |      0.0814679 |
| AL_81,AL_82,AL_83    |  0.0460322 |   0.00841911 |   0.184847  |      0.0797659 |
| AL_84,AL_85,AL_86    |  0.0857358 |   0.0414214  |   0.103723  |      0.0769601 |
| AL_168,AL_169,AL_170 |  0.0759808 | nan          | nan         |      0.0759808 |
| AL_189,AL_190,AL_191 |  0.0709249 | nan          | nan         |      0.0709249 |

High cross-group correlations between EK columns and engineered AL aggregates:

| feature_1   | feature_2     |   spearman_rho |
|:------------|:--------------|---------------:|
| EK_3        | EK7xEK9       |       0.614823 |
| EK_7        | EK7xEK9       |       0.952182 |
| EK_9        | EK7xEK9       |       0.893963 |
| EK_9        | EK7_minus_EK9 |      -0.558858 |

AL PCA top components:

| component   |   explained_variance_ratio |   cumulative_explained_variance |
|:------------|---------------------------:|--------------------------------:|
| PC1         |                 0.307286   |                        0.307286 |
| PC2         |                 0.106777   |                        0.414063 |
| PC3         |                 0.0242552  |                        0.438318 |
| PC4         |                 0.0240037  |                        0.462322 |
| PC5         |                 0.0229102  |                        0.485232 |
| PC6         |                 0.0185244  |                        0.503756 |
| PC7         |                 0.0122224  |                        0.515979 |
| PC8         |                 0.00969636 |                        0.525675 |
| PC9         |                 0.00948092 |                        0.535156 |
| PC10        |                 0.00887154 |                        0.544027 |
| PC11        |                 0.00872562 |                        0.552753 |
| PC12        |                 0.00760385 |                        0.560357 |
| PC13        |                 0.00665249 |                        0.567009 |
| PC14        |                 0.0052269  |                        0.572236 |
| PC15        |                 0.00499033 |                        0.577227 |
| PC16        |                 0.00471491 |                        0.581942 |
| PC17        |                 0.00452263 |                        0.586464 |
| PC18        |                 0.00427836 |                        0.590743 |
| PC19        |                 0.00426004 |                        0.595003 |
| PC20        |                 0.00417322 |                        0.599176 |

Figure: `reports\eda\figures\al_pca_cumulative_variance.png`

Leakage scan:

None above 0.85.

## Phase 5 - Cross-Dataset Comparison

Top-feature distribution shifts by dataset:

| dataset   | feature      |          mean |        median |   pathogenic_mean |   benign_mean |   missing_rate |
|:----------|:-------------|--------------:|--------------:|------------------:|--------------:|---------------:|
| MASTER    | EK_7         |   5.94423     |   7.10006     |       6.63671     |   3.84944     |      0.122484  |
| MASTER    | EK7xEK9      |  54.2013      |  55.0899      |      61.3851      |  32.7486      |      0.207097  |
| MASTER    | EK_9         |   7.48213     |   7.85384     |       8.20932     |   5.31054     |      0.207097  |
| MASTER    | EK_3         |   3.0462      |   3.69173     |       3.52424     |   1.89586     |      0.45377   |
| MASTER    | n_pops       |  10.32        |  10           |       8.63285     |  14.9565      |      0         |
| MASTER    | EK_2         |   4.46425     |   5.09838     |       4.85101     |   3.2943      |      0.122484  |
| MASTER    | n_nonmiss_AL | 144.114       | 125           |     126.291       | 193.092       |      0         |
| MASTER    | AL_287       |   0.0146874   |   0.000745476 |       0.0109631   |   0.0200329   |      0.665984  |
| MASTER    | AL_215       |   0.0176612   |   0.000985283 |       0.0136727   |   0.0220978   |      0.743432  |
| MASTER    | AL_26        |   0.0147791   |   0.00086996  |       0.0101997   |   0.0245036   |      0.370181  |
| KANSER    | EK_7         |   5.43696     |   5.86898     |       6.46119     |   2.83163     |      0.0592784 |
| KANSER    | EK7xEK9      |  49.8612      |  51.2409      |      60.4996      |  22.5352      |      0.0618557 |
| KANSER    | EK_9         |   7.35106     |   7.86768     |       8.64        |   4.04026     |      0.0618557 |
| KANSER    | EK_3         |   3.4326      |   4.05306     |       4.21265     |   1.94263     |      0.332474  |
| KANSER    | n_pops       |   9.39948     |  10           |       5.44776     |  18.225       |      0         |
| KANSER    | EK_2         |   4.40469     |   5.21523     |       5.07002     |   2.7123      |      0.0592784 |
| KANSER    | n_nonmiss_AL | 135.335       |  88.5         |      96.4776      | 222.117       |      0         |
| KANSER    | AL_287       |   0.0192075   |   0.00309741  |       0.0292408   |   0.0134742   |      0.716495  |
| KANSER    | AL_215       |   0.0189814   |   0.000940006 |       0.0347636   |   0.0137207   |      0.773196  |
| KANSER    | AL_26        |   0.0128038   |   0.000815929 |       0.0133647   |   0.0122327   |      0.427835  |
| PAH       | EK_7         |   6.14113     |   6.98214     |       6.51269     |   4.15949     |      0.0295699 |
| PAH       | EK7xEK9      |  56.476       |  56.3422      |      60.434       |  35.3671      |      0.0295699 |
| PAH       | EK_9         |   7.60016     |   7.9398      |       7.93108     |   5.83527     |      0.0295699 |
| PAH       | EK_3         |   2.54227     |   3.4801      |       2.74056     |   1.77885     |      0.478495  |
| PAH       | n_pops       |   9.46505     |  10           |       9.5129      |   9.22581     |      0         |
| PAH       | EK_2         |   4.67046     |   5.44631     |       4.87549     |   3.57695     |      0.0295699 |
| PAH       | n_nonmiss_AL | 145.159       | 135           |     144.832       | 146.79        |      0         |
| PAH       | AL_287       |   0.00023978  |   6.71712e-05 |       0.000231468 |   0.000287805 |      0.672043  |
| PAH       | AL_215       |   0.000275182 |   0.000125767 |       0.000265838 |   0.000326933 |      0.771505  |
| PAH       | AL_26        |   0.000353503 |   6.26155e-05 |       0.000317547 |   0.000536047 |      0.362903  |
| CFTR      | EK_7         |   6.71004     |   7.41467     |       7.26126     |   4.34771     |      0         |
| CFTR      | EK7xEK9      |  62.3087      |  59.0575      |      69.575       |  31.1673      |      0         |
| CFTR      | EK_9         |   8.28452     |   8.09966     |       9.03258     |   5.07854     |      0         |
| CFTR      | EK_3         |   3.14178     |   3.45787     |       3.382       |   2.5059      |      0.441441  |
| CFTR      | n_pops       |  14.2432      |  10           |      13.8444      |  15.9524      |      0         |
| CFTR      | EK_2         |   5.04764     |   5.43089     |       5.21396     |   4.33486     |      0         |
| CFTR      | n_nonmiss_AL | 229.018       | 299           |     227.3         | 236.381       |      0         |
| CFTR      | AL_287       |   0.00883487  |   0.00037104  |       0.00769573  |   0.01331     |      0.378378  |
| CFTR      | AL_215       |   0.0184161   |   0.00360347  |       0.0168728   |   0.0239463   |      0.504505  |
| CFTR      | AL_26        |   0.0172779   |   0.00295267  |       0.0176147   |   0.0156148   |      0.144144  |

Panel-unique versus shared variants:

| panel   | subset             |   n |   pathogenic_rate |   overall_missing_rate |   numeric_range_mean |
|:--------|:-------------------|----:|------------------:|-----------------------:|---------------------:|
| KANSER  | unique             | 142 |          0.316901 |               0.427742 |             0.647921 |
| KANSER  | shared_with_master | 246 |          0.906504 |               0.654875 |             0.828301 |
| PAH     | unique             | 117 |          0.581197 |               0.576814 |             0.487153 |
| PAH     | shared_with_master | 255 |          0.94902  |               0.527445 |             0.531461 |
| CFTR    | unique             |  34 |          0.5      |               0.224546 |             0.552333 |
| CFTR    | shared_with_master |  77 |          0.948052 |               0.337331 |             0.554045 |

KANSER differs from MASTER primarily through a much lower panel-unique pathogenic rate and a cancer-panel-specific distribution of computational and population-frequency features. PAH and CFTR panel-unique subsets are smaller and have intermediate pathogenic rates, so their estimates are less stable.

## Phase 6 - ACMG Evidence Interpretation

Strong features mapped to plausible ACMG evidence categories. These mappings are inferred from statistics and biological plausibility, not confirmed metadata.

| feature      |   auc_equivalent | proposed_acmg_category   | justification                                                                                      | confidence   |
|:-------------|-----------------:|:-------------------------|:---------------------------------------------------------------------------------------------------|:-------------|
| EK_7         |         0.738042 | PP3/BP4                  | Computational/evolutionary pathogenicity evidence.                                                 | medium       |
| EK7xEK9      |         0.726059 | PP3/BP4                  | Computational/evolutionary pathogenicity evidence.                                                 | medium       |
| EK_9         |         0.695513 | PP3/BP4                  | Computational/evolutionary pathogenicity evidence.                                                 | medium       |
| EK_3         |         0.690208 | PP3/BP4                  | Computational/evolutionary pathogenicity evidence.                                                 | medium       |
| n_pops       |         0.316095 | PM2/BS2 proxy            | Database absence/presence aggregate; missingness encodes rarity and ascertainment.                 | high         |
| EK_2         |         0.67377  | PP3/BP4                  | Computational/evolutionary pathogenicity evidence.                                                 | medium       |
| n_nonmiss_AL |         0.336843 | PM2/BS2 proxy            | Database absence/presence aggregate; missingness encodes rarity and ascertainment.                 | high         |
| AL_287       |         0.342764 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_215       |         0.35167  | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_26        |         0.355282 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| EK_4         |         0.637392 | PP3/BP4                  | Computational/evolutionary pathogenicity evidence.                                                 | medium       |
| AL_186       |         0.366538 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_251       |         0.36908  | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_73        |         0.369644 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| max_AF       |         0.372088 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_94        |         0.373917 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_79        |         0.374037 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_184       |         0.379006 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_327       |         0.38091  | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_136       |         0.386709 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_224       |         0.390054 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| EK_6         |         0.60949  | PP3/BP4                  | Computational/evolutionary pathogenicity evidence.                                                 | medium       |
| AL_91        |         0.393959 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_260       |         0.394473 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |
| AL_178       |         0.398848 | BA1/BS1/PM2              | Population frequency or rarity evidence; direction must be interpreted from AUC and distributions. | medium       |

Features that do not cleanly map to ACMG are marked as technical/statistical artifact candidates, especially PCA or availability-derived signals whose biology is indirect.
