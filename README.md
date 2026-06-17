# TEKNOFEST 2026 Genetic Variant Classifier

This repository implements the competition solution described in
`TEKNOFEST2026_FINAL_ANALYSIS_AND_MASTER_PROMPT.md`.

The first implemented slice is Part IV, Section B: data loading, overlap maps,
panel-unique splits, and dropping `AL_185`.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run First Section

```powershell
$env:PYTHONPATH="src"
python scripts/prepare_first_section.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main
```

Outputs are written to `data/processed`.

## Run Second Section

```powershell
$env:PYTHONPATH="src"
python scripts/engineer_second_section.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main
```

This fits the missingness PCA and target encoders on MASTER, then transforms the
panel datasets and panel-unique splits.

## Run Third Section

```powershell
$env:PYTHONPATH="src"
python scripts/build_validation_strategy.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main
```

This writes contamination-aware MASTER validation folds and fold summaries for
the model-training stage.

## Run Fourth Section

```powershell
$env:PYTHONPATH="src"
python scripts/train_fourth_section.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main --trials 100
```

This runs the ACMG rule baseline, EK-only logistic regression baseline,
LightGBM with Optuna, ExtraTrees comparison, and panel-unique LightGBM
evaluation. Final LightGBM artifacts are written to `models`.

## Run Fifth Section

```powershell
$env:PYTHONPATH="src"
python scripts/build_fifth_section_explainability.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main
```

This writes SHAP global importance, grouped ACMG importance, beeswarm and
dependence plots, representative waterfall cases, and the ACMG feature mapping
table to `reports/explainability`.

## Run PDF EDA Prompt

```powershell
$env:PYTHONPATH="src"
python scripts/run_pdf_eda_prompt.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main
```

This applies the six-phase EDA prompt from `Adsiz dokuman.pdf` and writes the
Markdown report, tables, and figures under `reports/eda`.

## Run Master Prompt Requirements

```powershell
$env:PYTHONPATH="src"
python scripts/run_master_prompt_requirements.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main --bootstrap 1000
```

This applies `prompt.pdf` end to end: overlap checks, leakage-safe feature
engineering verification, contamination-aware validation artifacts, 10 ablations,
panel-unique predictions with probabilities and binary labels, bootstrap CIs,
McNemar/statistical-test notes, calibration curves, and the implementation
checklist/report under `reports/master_prompt`.

## Resume Or Extend Full Optuna Study

```powershell
$env:PYTHONPATH="src"
python scripts/run_optuna_lgbm_study.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main --n-trials 50 --max-estimators 3000 --timeout-seconds 2400
```

The required full Optuna run has been completed with 101 complete trials. The
study remains resumable through `reports/master_prompt/optuna_lgbm_study.sqlite3`.
It writes `lgbm_optuna_trials_resumable.csv`, `lgbm_best_params_resumable.json`,
and `lgbm_optuna_convergence.png`.

## Run Competition-Ready Repository Pipeline

```powershell
python run_pipeline.py --mode smoke
python run_pipeline.py --mode full
python run_pipeline.py --mode tune
python run_pipeline.py --mode evaluate
```

The smoke mode is intended for quick verification and writes the same output
contract as full mode while using lighter bootstrap/training settings when
cached master-prompt outputs are absent. The pipeline writes:

- tables to `reports/tables`
- figures to `reports/figures`
- model/preprocessor/prediction artifacts to `artifacts`
- the final summary to `reports/final_model_report_summary.md`

Use `--mode tune` to run the Phase 9.6 performance-diagnosis layer after the
pipeline, creating experiment folders, comparison tables, final selection
artifacts, and `reports/final_performance_analysis.md`. Use `--mode evaluate`
to regenerate evaluation/diagnosis outputs from existing saved predictions and
model artifacts without retraining.

The current final interpretation is intentionally conservative: the model is
moderate, not weak and not strong. ROC-AUC and panel-unique generalization are
acceptable, while MASTER F1-macro and MCC remain moderate-to-good. The surgical
Phase 10 diagnostics focus on threshold stability, MCC-aware thresholding,
probability calibration, overfitting gaps, panel-specific errors, controlled
LightGBM profiles, and feature-ablation evidence. Key outputs include:

- `reports/tables/advanced_threshold_comparison.csv`
- `reports/tables/fold_threshold_stability.csv`
- `reports/tables/calibration_comparison.csv`
- `reports/tables/overfitting_gap_analysis.csv`
- `reports/tables/feature_group_ablation_results.csv`
- `reports/tables/panel_specific_error_analysis.csv`
- `reports/tables/final_model_selection_table.csv`
- `reports/current_results_verification.md`

The final scientific audit/report-readiness pass additionally writes:

- `reports/final_metric_verification_audit.md`
- `reports/tables/phase10_before_after_comparison.csv`
- `reports/calibration_decision_review.md`
- `reports/tables/final_error_group_feature_summary.csv`
- `reports/tables/final_panel_specific_metrics.csv`
- `reports/final_feature_interpretation.md`
- `reports/final_model_strength_statement.md`

Required notebooks are generated with:

```powershell
python scripts/create_required_notebooks.py
```

They are saved under `notebooks/`:

- `01_data_understanding.ipynb`
- `02_feature_engineering_and_validation.ipynb`
- `03_model_training_and_evaluation.ipynb`
- `04_explainability_and_report_outputs.ipynb`

Run verification tests with:

```powershell
python -m pytest tests
python src/smoke_test.py
```

## Run PHASE 9.5 Visualizations And Metrics

```powershell
python scripts/run_phase9_outputs.py
```

This reads the existing model predictions and writes the official-report
visualizations and general metric table required by `PHASE 9.pdf`:

- `reports/tables/all_evaluation_metrics.csv`
- `reports/figures/correlation_matrix_top_features.png`
- split-specific confusion matrices
- MASTER and panel ROC/PR curves
- `threshold_optimization.png`
- `model_comparison_metrics.png`
- `feature_importance_top30.png`
- class distribution, missingness, and error-analysis plots
