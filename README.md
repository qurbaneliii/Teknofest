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
