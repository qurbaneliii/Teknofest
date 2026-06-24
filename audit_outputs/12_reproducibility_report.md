# Reproducibility

Python/package pins are in `requirements.txt`.

- Audit and train audited candidate: `python scripts/run_teknofest_readiness_audit.py`
- Tests: `python -m pytest tests -q`
- Inference: `python scripts/generate_final_submission.py --input-csv path/to/test.csv --output outputs/submission.csv` (pass `--decision` after the CLI update if using audited decision)

Seed: 42. Raw data are not modified.
