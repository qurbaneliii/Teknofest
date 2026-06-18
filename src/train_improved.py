from __future__ import annotations

from pathlib import Path
import sys

from config import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loading import discover_data_dir
from phase10_improvements import run_phase10_improvements
from run_pipeline import load_params
from teknofest.data_prep import prepare_data


def train_improved(mode: str = "evaluate", data_dir: Path | None = None) -> dict[str, object]:
    """Run the model-performance improvement workflow on existing predictions/artifacts."""
    resolved_data_dir = data_dir or discover_data_dir(PROJECT_ROOT)
    prepared = prepare_data(resolved_data_dir)
    return run_phase10_improvements(prepared, load_params(), mode=mode)


__all__ = ["train_improved"]
