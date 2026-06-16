from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from teknofest.data_prep import prepare_data
from teknofest.training import optimize_lgbm_resumable


DEFAULT_DATA_DIR = Path("teknofest2026_artificialintelligenceinhealtcare-main")


def save_convergence_plot(trials: pd.DataFrame, out_path: Path) -> None:
    complete = trials[trials["state"] == "COMPLETE"].copy()
    if complete.empty:
        return
    complete = complete.sort_values("trial")
    complete["best_auc_so_far"] = complete["mean_auc"].cummax()
    plt.figure(figsize=(8, 5))
    plt.plot(complete["trial"], complete["mean_auc"], marker="o", linewidth=1, label="Trial AUC")
    plt.plot(complete["trial"], complete["best_auc_so_far"], linewidth=2, label="Best so far")
    plt.xlabel("Trial")
    plt.ylabel("Mean contamination-aware CV AUC")
    plt.title("LightGBM Optuna convergence")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/resume the LightGBM Optuna study from prompt.pdf.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/master_prompt"))
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-estimators", type=int, default=3000)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--study-name", default="teknofest_lgbm")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{(args.out_dir / 'optuna_lgbm_study.sqlite3').resolve().as_posix()}"
    prepared = prepare_data(args.data_dir)
    best_params, trials = optimize_lgbm_resumable(
        prepared,
        n_trials=args.n_trials,
        storage_url=storage_url,
        study_name=args.study_name,
        max_estimators=args.max_estimators,
        timeout_seconds=args.timeout_seconds,
    )
    trials.to_csv(args.out_dir / "lgbm_optuna_trials_resumable.csv", index=False)
    (args.out_dir / "lgbm_best_params_resumable.json").write_text(
        json.dumps(best_params, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_convergence_plot(trials, args.out_dir / "lgbm_optuna_convergence.png")
    complete_trials = int((trials["state"] == "COMPLETE").sum())
    running_trials = int((trials["state"] == "RUNNING").sum())
    print(f"Complete trials: {complete_trials}")
    print(f"Running trials: {running_trials}")
    print(f"Best params written to: {(args.out_dir / 'lgbm_best_params_resumable.json').resolve()}")


if __name__ == "__main__":
    main()
