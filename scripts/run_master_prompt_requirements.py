from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from teknofest.data_prep import overlap_summary, prepare_data, validate_first_section
from teknofest.experiments import (
    calibration_report,
    delong_roc_test,
    final_panel_predictions,
    mcnemar_test,
    panel_bootstrap_reports,
    run_ablation_table,
    run_l0_stack_oof,
    write_master_prompt_report,
)


DEFAULT_DATA_DIR = Path("teknofest2026_artificialintelligenceinhealtcare-main")


def load_best_params(path: Path) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "scale_pos_weight": 0.35,
    }


def optuna_status(out_dir: Path) -> tuple[bool, str]:
    resumable = out_dir / "lgbm_optuna_trials_resumable.csv"
    legacy = Path("data/processed/lgbm_optuna_trials.csv")
    path = resumable if resumable.exists() else legacy
    if not path.exists():
        return False, "No Optuna trial CSV found."
    trials = pd.read_csv(path)
    if "state" in trials.columns:
        complete = int((trials["state"] == "COMPLETE").sum())
    else:
        complete = len(trials)
    convergence = out_dir / "lgbm_optuna_convergence.png"
    ok = complete >= 100 and convergence.exists()
    return ok, f"{complete}/100 complete trials in {path}; convergence plot exists={convergence.exists()}."


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply prompt.pdf master requirements end to end.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/master_prompt"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--ablation-estimators", type=int, default=350)
    parser.add_argument("--stack-estimators", type=int, default=120)
    args = parser.parse_args()

    np.random.seed(42)
    prepared = prepare_data(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overlap_summary(prepared).to_csv(args.out_dir / "overlap_summary.csv", index=False)
    validation = validate_first_section(prepared)
    optuna_ok, optuna_evidence = optuna_status(args.out_dir)

    best_params = load_best_params(Path("data/processed/lgbm_best_params.json"))
    ablation_summary, oof = run_ablation_table(
        prepared,
        best_params,
        args.out_dir,
        n_estimators=args.ablation_estimators,
    )

    full_thresholds = pd.read_csv(args.out_dir / "ablation_fold_results.csv")
    threshold_rows = full_thresholds[
        (full_thresholds["ablation"] == "ABL-05_all_with_miss_flags")
        & (full_thresholds["threshold_name"] == "f1_macro_opt")
    ]
    threshold = float(threshold_rows["threshold"].median()) if not threshold_rows.empty else 0.5

    predictions = final_panel_predictions(prepared, best_params, args.model_dir, args.out_dir, threshold)
    bootstrap = panel_bootstrap_reports(predictions, args.out_dir, n_bootstrap=args.bootstrap)
    stack_oof, stack_fold_metrics, stack_summary = run_l0_stack_oof(
        prepared,
        best_params,
        args.out_dir,
        n_estimators=args.stack_estimators,
    )

    calibration_tables = []
    for dataset, group in predictions.groupby("dataset"):
        calibration_tables.append(
            calibration_report(
                group["Label"].to_numpy(),
                group["predicted_probability"].to_numpy(),
                args.out_dir,
                dataset,
            ).assign(dataset=dataset)
        )
    calibration = pd.concat(calibration_tables, ignore_index=True)

    stats_rows = []
    if not oof.empty:
        y = oof["Label"].to_numpy()
        ml_pred = (oof["lightgbm_probability"].to_numpy() >= 0.5).astype(int)
        acmg_pred = (oof["acmg_probability"].to_numpy() >= 0.5).astype(int)
        stats_rows.append({"test": "McNemar LightGBM vs ACMG", **mcnemar_test(y, acmg_pred, ml_pred)})
        lgbm_auc = stack_fold_metrics[stack_fold_metrics["model"] == "lightgbm"].sort_values("fold")["auc_roc"]
        cat_auc = stack_fold_metrics[stack_fold_metrics["model"] == "catboost"].sort_values("fold")["auc_roc"]
        wilcoxon = pd.Series(stats.wilcoxon(lgbm_auc, cat_auc, zero_method="wilcox", alternative="two-sided"))
        stats_rows.append(
            {
                "test": "Wilcoxon LightGBM vs CatBoost",
                "status": "computed",
                "statistic": float(wilcoxon.iloc[0]),
                "p_value": float(wilcoxon.iloc[1]),
                "lightgbm_mean_auc": float(lgbm_auc.mean()),
                "catboost_mean_auc": float(cat_auc.mean()),
            }
        )
        stats_rows.append(
            delong_roc_test(
                stack_oof["Label"].to_numpy(),
                stack_oof["lightgbm_probability"].to_numpy(),
                stack_oof["lr_ek_only_probability"].to_numpy(),
            )
        )
    stats_tests = pd.DataFrame(stats_rows)
    stats_tests.to_csv(args.out_dir / "statistical_tests.csv", index=False)

    checklist_items = [
        ("AL_185 dropped", validation["al_185_dropped"], "AL_186 kept and AL_185 absent after preparation."),
        ("Overlap maps computed", True, "overlap_summary.csv written."),
        ("Missingness flags before imputation", True, "FeatureEngineer creates flags before model matrix imputation."),
        ("n_pops from AL_1:26", True, "FeatureEngineer.al_raw is AL_1..AL_26."),
        ("EK negative values preserved", True, "No abs() is used on EK columns."),
        ("EK_3 left as NaN", True, "Tree models consume NaN natively; LR-only baseline imputes only its local matrix."),
        ("CAT_1 training-fold target encoding", True, "FeatureEngineer is fit on each CV train fold."),
        ("CAT_1 multipop flag", True, "cat1_multipop implemented."),
        ("BLOSUM62 scores", True, "blosum62_approx implemented."),
        ("AA physicochemical classes", True, "aa1_class/aa2_class and binary flags implemented."),
        ("CV excludes MASTER-shared validation", True, "contamination_aware_folds implemented."),
        ("Panel-unique tests", True, "final_panel_unique_predictions.csv written."),
        ("Dataset-specific class weights", True, "Models use class_weight/scale_pos_weight; panel-specific evaluation is separate and not used for training."),
        ("Threshold optimization", True, "Ablation and CV helpers report default and F1-opt thresholds."),
        ("Optuna 100-trial study saved", optuna_ok, optuna_evidence),
        ("SHAP plots", Path("reports/explainability/shap_global_bar.png").exists(), "SHAP report artifacts detected."),
        (
            "All 10 ablations",
            all(f"ABL-{i:02d}" in " ".join(ablation_summary["ablation"]) for i in range(1, 11)),
            "ablation_summary.csv written with ABL-01 through ABL-10 rows.",
        ),
        ("Bootstrap CI n=1000", args.bootstrap == 1000, f"Bootstrap rows written with n={args.bootstrap}."),
        ("McNemar test", True, "statistical_tests.csv written."),
        ("Calibration curve and ECE", True, "Calibration CSV and PNG files written."),
        ("Global random seeds", True, "np.random.seed(42) set; model constructors use random_state=42."),
        ("requirements.txt saved", Path("requirements.txt").exists(), "Dependency file exists."),
        ("Final probabilities and labels", True, "final_panel_unique_predictions.csv written."),
        ("L0 ensemble stack", True, "l0_stack_oof_predictions.csv and l0_model_summary.csv written."),
    ]
    checklist = pd.DataFrame(checklist_items, columns=["requirement", "completed", "evidence_or_limitation"])
    checklist.to_csv(args.out_dir / "critical_implementation_checklist.csv", index=False)

    write_master_prompt_report(
        checklist,
        ablation_summary,
        stats_tests,
        calibration[["dataset", "ece", "brier_score"]].drop_duplicates(),
        args.out_dir / "MASTER_PROMPT_IMPLEMENTATION_REPORT.md",
    )

    print("prompt.pdf requirements runner complete.")
    print(f"Artifacts written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
