from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "teknofest2026_artificialintelligenceinhealtcare-main"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSORS_DIR = ARTIFACTS_DIR / "preprocessors"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
RANDOM_SEED = 42

