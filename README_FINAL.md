# TEKNOFEST Audited Final Pipeline

This is the competition-facing, audited variant pathogenicity classifier. The
final model is `audited_master_only_lgbm`; its threshold is saved in
`artifacts/metrics/audited_final_model_decision.json`.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Place the supplied organizer training CSVs under
`teknofest2026_artificialintelligenceinhealtcare-main/`.

## Reproduce final evidence

```powershell
python src/run_final_pipeline.py
python src/sanity_check_final.py
```

This regenerates fold-safe MASTER-only OOF evidence, panel-unique evaluations,
metrics, threshold sweep, figures, final model artifact, and PDR assets.

## Predict official test data

```powershell
python src/predict_final.py --input path/to/official_test.csv --output outputs/final/submission.csv --basic-submission
```

`Label`, if present, is ignored. The official submission schema was not present
in this repository, so confirm its required column names before submission.

## Outputs and limitations

Use `outputs/final/TEKNOFEST_MODEL_READINESS_REPORT.md` as the authoritative
final report. The result is a Conditional Go: official test distribution and
submission-template compliance cannot be verified locally.
