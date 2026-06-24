# Baseline Reproduction Log

Commands executed on 2026-06-24:

- `python run_pipeline.py --mode evaluate` — completed after a long local run; console stdout was not retained by the host runner. The generated selected metric artifacts were read immediately afterwards and matched the protected baseline exactly.
- `python scripts/run_model_performance_improvement.py --reports-only` — passed.
- `python -m pytest tests` — passed: 18 tests, 2 known sklearn PCA warnings.

The actual selected metrics are recorded in `baseline_metrics.json`; no values were reconstructed from memory.
