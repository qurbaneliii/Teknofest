from __future__ import annotations

import argparse
import json
from pathlib import Path

from teknofest.data_prep import prepare_data
from teknofest.training import (
    evaluate_panel_unique,
    fit_final_lgbm,
    optimize_lgbm,
    run_cv_baselines,
)


DEFAULT_DATA_DIR = Path("teknofest2026_artificialintelligenceinhealtcare-main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Part IV, Section E model training.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    prepared = prepare_data(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_params, trials = optimize_lgbm(prepared, n_trials=args.trials)
    trials.to_csv(args.out_dir / "lgbm_optuna_trials.csv", index=False)
    (args.out_dir / "lgbm_best_params.json").write_text(
        json.dumps(best_params, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    cv_results = run_cv_baselines(prepared, lgbm_params=best_params)
    cv_results.to_csv(args.out_dir / "model_training_cv_results.csv", index=False)
    cv_summary = (
        cv_results.groupby(["model", "threshold_name"])[["auc_roc", "auc_pr", "f1_macro", "mcc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    cv_summary.to_csv(args.out_dir / "model_training_cv_summary.csv", index=False)

    engineer, model, columns = fit_final_lgbm(prepared, best_params, args.model_dir)
    panel_results = evaluate_panel_unique(prepared, engineer, model, columns)
    panel_results.to_csv(args.out_dir / "panel_unique_lightgbm_results.csv", index=False)

    print("Fourth section model training complete.")
    print(f"Optuna trials: {args.trials}")
    print(f"Best LightGBM params: {best_params}")
    print(cv_summary.to_string(index=False))
    print(panel_results.to_string(index=False))
    print(f"Models written to: {args.model_dir.resolve()}")


if __name__ == "__main__":
    main()
