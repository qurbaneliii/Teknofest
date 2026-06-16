from __future__ import annotations

from pathlib import Path

from teknofest.data_prep import PreparedData
from teknofest.experiments import run_l0_stack_oof
from teknofest.training import fit_final_lgbm, optimize_lgbm_resumable


def train_final_lightgbm(prepared: PreparedData, params: dict[str, object], model_dir: str | Path):
    return fit_final_lgbm(prepared, params, Path(model_dir))


def run_optional_stack(prepared: PreparedData, params: dict[str, object], out_dir: str | Path, n_estimators: int = 60):
    return run_l0_stack_oof(prepared, params, Path(out_dir), n_estimators=n_estimators)


def run_resumable_optuna(prepared: PreparedData, n_trials: int, storage_url: str):
    return optimize_lgbm_resumable(prepared, n_trials=n_trials, storage_url=storage_url)

