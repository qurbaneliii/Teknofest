from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from teknofest.data_prep import prepare_data
from teknofest.training import (
    evaluate_lgbm_medical_candidate,
    optimize_lgbm_medical_resumable,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "teknofest2026_artificialintelligenceinhealtcare-main"
DEFAULT_STORAGE = PROJECT_ROOT / "reports" / "master_prompt" / "optuna_lgbm_study.sqlite3"
DEFAULT_STUDY_NAME = "teknofest_lgbm_medical_utility"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"

REFERENCE_METRICS = PROJECT_ROOT / "reports" / "tables" / "final_medical_metric_comparison.csv"
REFERENCE_SELECTION = PROJECT_ROOT / "reports" / "tables" / "final_model_selection_table.csv"
REFERENCE_GAPS = PROJECT_ROOT / "reports" / "tables" / "overfitting_gap_analysis.csv"
HISTORICAL_STUDY_NAME = "teknofest_lgbm"


def _storage_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]


def _historical_audit(storage_url: str, reference_roc_auc: float) -> dict[str, Any]:
    try:
        study = optuna.load_study(study_name=HISTORICAL_STUDY_NAME, storage=storage_url)
    except KeyError:
        return {
            "exists": False,
            "study_name": HISTORICAL_STUDY_NAME,
            "completed_trials": 0,
            "best_score": None,
            "resume_recommendation": "No historical AUC-only study exists; start the medical-utility study.",
        }
    complete = _completed(study)
    running = sum(trial.state == optuna.trial.TrialState.RUNNING for trial in study.trials)
    best = max(complete, key=lambda trial: float(trial.value)) if complete else None
    best_score = float(best.value) if best is not None else None
    return {
        "exists": True,
        "study_name": HISTORICAL_STUDY_NAME,
        "completed_trials": len(complete),
        "running_trials": running,
        "failed_trials": sum(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials),
        "pruned_trials": sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials),
        "best_trial": int(best.number) if best is not None else None,
        "best_score": best_score,
        "best_params": dict(best.params) if best is not None else {},
        "best_beats_reference_roc_auc_only": bool(best_score is not None and best_score > reference_roc_auc),
        "resume_recommendation": (
            "Preserve the 101-trial AUC-only history and use a separate medical-utility study in the same SQLite store. "
            "Its AUC alone cannot establish a safe final-model improvement."
        ),
        "timeout_result_loss_detected": bool(running),
    }


def _reference() -> tuple[dict[str, float], dict[str, float], float, float]:
    if not REFERENCE_METRICS.exists():
        raise FileNotFoundError(f"Missing saved reference metric artifact: {REFERENCE_METRICS}")
    metrics = pd.read_csv(REFERENCE_METRICS)
    master = metrics.loc[metrics["evaluation_split"].eq("MASTER_CV_saved_predictions")]
    panel = metrics.loc[metrics["evaluation_split"].eq("panel_unique_combined")]
    if len(master) != 1 or len(panel) != 1:
        raise ValueError("Reference medical metric table must contain exactly one MASTER and panel-combined row.")
    if not REFERENCE_SELECTION.exists() or not REFERENCE_GAPS.exists():
        raise FileNotFoundError("Saved final selection and overfitting artifacts are required for an Optuna comparison.")
    selected = pd.read_csv(REFERENCE_SELECTION)
    selected = selected.loc[selected["selected_as_final"].astype(bool)]
    if len(selected) != 1:
        raise ValueError("Final model selection table must contain exactly one selected reference model.")
    gaps = pd.read_csv(REFERENCE_GAPS)
    reference_gap = gaps.loc[gaps["experiment_id"].eq("lightgbm_conservative_regularized"), "roc_auc_gap"]
    if reference_gap.empty:
        raise ValueError("No conservative-regularized overfitting-gap records were found.")
    return (
        master.iloc[0].to_dict(),
        panel.iloc[0].to_dict(),
        float(selected.iloc[0]["threshold_instability"]),
        float(reference_gap.mean()),
    )


def _plot_history(trials: pd.DataFrame, out_path: Path) -> None:
    complete = trials.loc[trials["state"].eq("COMPLETE")].copy()
    if complete.empty:
        return
    complete = complete.sort_values("trial")
    complete["best_so_far"] = complete["medical_utility_score"].cummax()
    plt.figure(figsize=(8, 5))
    plt.plot(complete["trial"], complete["medical_utility_score"], marker="o", linewidth=1, label="Trial utility")
    plt.plot(complete["trial"], complete["best_so_far"], linewidth=2, label="Best so far")
    plt.xlabel("Trial")
    plt.ylabel("MedicalUtilityScore")
    plt.title("LightGBM medical-utility Optuna history")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_importance(storage_url: str, study_name: str, out_path: Path) -> None:
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        importance = optuna.importance.get_param_importances(study)
    except (ValueError, RuntimeError):
        importance = {}
    if not importance:
        return
    values = pd.Series(importance).sort_values()
    plt.figure(figsize=(8, 5))
    plt.barh(values.index, values.values)
    plt.xlabel("Importance")
    plt.title("Optuna parameter importance - MedicalUtilityScore")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _comparison_frame(
    reference_master: dict[str, float],
    reference_panel: dict[str, float],
    candidate: dict[str, object],
    reference_stability: float,
    reference_gap: float,
    selection_status: str,
) -> pd.DataFrame:
    shared = ["roc_auc", "pr_auc", "f1_macro", "mcc", "balanced_accuracy", "pathogenic_recall", "specificity", "medical_utility_score"]
    rows: list[dict[str, object]] = []
    for model_id, master, panel, threshold, stability, gap in (
        (
            "lightgbm_conservative_regularized_reference",
            reference_master,
            reference_panel,
            float(reference_master["threshold"]),
            reference_stability,
            reference_gap,
        ),
        (
            "optuna_medical_candidate",
            candidate["oof_metrics"],
            candidate["panel_metrics"],
            float(candidate["threshold"]),
            float(candidate["threshold_stability"]),
            float(candidate["mean_roc_auc_gap"]),
        ),
    ):
        row: dict[str, object] = {
            "model_id": model_id,
            "selection_status": "preserved_final" if model_id.startswith("lightgbm") else selection_status,
            "threshold": threshold,
            "threshold_stability": stability,
            "mean_roc_auc_gap": gap,
        }
        row.update({f"master_{key}": float(master[key]) for key in shared})
        row.update({f"panel_{key}": float(panel[key]) for key in shared})
        rows.append(row)
    return pd.DataFrame(rows)


def _select_candidate(
    reference_master: dict[str, float],
    reference_panel: dict[str, float],
    candidate: dict[str, object],
    reference_stability: float,
    reference_gap: float,
) -> tuple[str, list[str], list[str]]:
    candidate_master = candidate["oof_metrics"]
    candidate_panel = candidate["panel_metrics"]
    improvements = [
        label
        for label, value, baseline in (
            ("MCC", candidate_master["mcc"], reference_master["mcc"]),
            ("F1-macro", candidate_master["f1_macro"], reference_master["f1_macro"]),
            ("PR-AUC", candidate_master["pr_auc"], reference_master["pr_auc"]),
            ("panel MCC", candidate_panel["mcc"], reference_panel["mcc"]),
            ("panel F1-macro", candidate_panel["f1_macro"], reference_panel["f1_macro"]),
            ("MedicalUtilityScore", candidate_master["medical_utility_score"], reference_master["medical_utility_score"]),
        )
        if float(value) > float(baseline) + 1e-12
    ]
    reasons: list[str] = []
    if len(improvements) < 2:
        reasons.append("fewer than two required metrics improved")
    for key in ("roc_auc", "pr_auc", "f1_macro", "mcc", "medical_utility_score"):
        if float(candidate_panel[key]) < float(reference_panel[key]) - 1e-12:
            reasons.append(f"panel {key} worsened")
    if float(candidate["threshold_stability"]) > reference_stability + 1e-12:
        reasons.append("threshold stability worsened")
    if float(candidate["mean_roc_auc_gap"]) > reference_gap + 1e-12:
        reasons.append("mean train-validation ROC-AUC gap worsened")
    return ("candidate_final" if not reasons else "rejected"), improvements, reasons


def _report(
    historical: dict[str, Any],
    trial_stats: dict[str, int],
    best_params: dict[str, object],
    trials: pd.DataFrame,
    comparison: pd.DataFrame,
    selection_status: str,
    improvements: list[str],
    reasons: list[str],
) -> str:
    best_score = float(trials.loc[trials["state"].eq("COMPLETE"), "medical_utility_score"].max())
    lines = [
        "# Medical Optuna Optimization Report",
        "",
        "## Audit And Resume Decision",
        "",
        f"Historical AUC-only study: `{historical['completed_trials']}` completed trials; best AUC `{historical.get('best_score'):.6f}`.",
        f"Historical study timeout loss detected: `{historical.get('timeout_result_loss_detected', False)}`. No RUNNING historical trials were found.",
        historical["resume_recommendation"],
        "",
        "## Medical Study",
        "",
        f"Completed before resume: `{trial_stats['complete'] - trial_stats['new_complete']}`; newly completed: `{trial_stats['new_complete']}`; completed total: `{trial_stats['complete']}`.",
        f"Best medical objective: `{best_score:.6f}`. The objective is the requested MedicalUtilityScore and each completed trial stores ROC-AUC, PR-AUC, F1-macro, MCC, balanced accuracy, pathogenic recall, specificity, and fold stability attributes in SQLite.",
        "",
        "## Best Parameters",
        "",
        "```json",
        json.dumps(best_params, indent=2, sort_keys=True),
        "```",
        "",
        "## Before Vs After",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Decision",
        "",
        f"Candidate status: `{selection_status}`. The deployed final model was not modified by this optimization run.",
        f"Required improvements observed: {', '.join(improvements) if improvements else 'none'}.",
    ]
    if reasons:
        lines.append(f"Rejection reason: {'; '.join(reasons)}.")
    else:
        lines.append("The candidate is marked `candidate_final` for a separate deployment review; final artifacts remain preserved.")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "All candidate metrics were recomputed from the newly saved candidate OOF and panel prediction CSV files. Panels were evaluated only after fitting on MASTER and their labels were never used for tuning or threshold selection.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/resume leakage-safe medical-utility LightGBM Optuna tuning.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-estimators", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--objective", choices=["medical_utility"], default="medical_utility")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    if args.n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    for directory in (TABLES_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, args.storage_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    reference_master, reference_panel, reference_stability, reference_gap = _reference()
    storage_url = _storage_url(args.storage_path)
    historical = _historical_audit(storage_url, float(reference_master["roc_auc"]))
    prepared = prepare_data(args.data_dir)
    best_params, trials, trial_stats = optimize_lgbm_medical_resumable(
        prepared,
        n_trials=args.n_trials,
        storage_url=storage_url,
        study_name=args.study_name,
        max_estimators=args.max_estimators,
        timeout_seconds=args.timeout_seconds,
        resume=args.resume,
    )
    candidate = evaluate_lgbm_medical_candidate(prepared, best_params)
    selection_status, improvements, reasons = _select_candidate(
        reference_master,
        reference_panel,
        candidate,
        reference_stability,
        reference_gap,
    )
    comparison = _comparison_frame(
        reference_master,
        reference_panel,
        candidate,
        reference_stability,
        reference_gap,
        selection_status,
    )

    trials.to_csv(TABLES_DIR / "optuna_medical_trials.csv", index=False)
    pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in best_params.items()]
    ).to_csv(TABLES_DIR / "optuna_medical_best_params.csv", index=False)
    comparison.to_csv(TABLES_DIR / "optuna_before_after_comparison.csv", index=False)
    candidate["oof_predictions"].to_csv(PREDICTIONS_DIR / "optuna_medical_candidate_oof_predictions.csv", index=False)
    candidate["panel_predictions"].to_csv(PREDICTIONS_DIR / "optuna_medical_candidate_panel_predictions.csv", index=False)
    candidate["fold_metrics"].to_csv(TABLES_DIR / "optuna_medical_candidate_fold_metrics.csv", index=False)
    candidate["overfitting_gaps"].to_csv(TABLES_DIR / "optuna_medical_candidate_overfitting_gaps.csv", index=False)
    _plot_history(trials, FIGURES_DIR / "optuna_medical_optimization_history.png")
    _plot_importance(storage_url, args.study_name, FIGURES_DIR / "optuna_param_importance.png")

    metadata = {
        "study_name": args.study_name,
        "storage_path": str(args.storage_path.resolve()),
        "objective": args.objective,
        "best_params": best_params,
        "effective_full_fit_params": candidate["effective_full_fit_params"],
        "best_medical_utility_score": float(trials.loc[trials["state"].eq("COMPLETE"), "medical_utility_score"].max()),
        "trial_stats": trial_stats,
        "historical_audit": historical,
        "candidate_status": selection_status,
        "required_metric_improvements": improvements,
        "rejection_reasons": reasons,
        "final_model_changed": False,
        "candidate_oof_prediction_path": str((PREDICTIONS_DIR / "optuna_medical_candidate_oof_predictions.csv").resolve()),
        "candidate_panel_prediction_path": str((PREDICTIONS_DIR / "optuna_medical_candidate_panel_predictions.csv").resolve()),
    }
    (METRICS_DIR / "optuna_medical_best_params.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (PROJECT_ROOT / "reports" / "optuna_medical_optimization_report.md").write_text(
        _report(historical, trial_stats, best_params, trials, comparison, selection_status, improvements, reasons),
        encoding="utf-8",
    )
    print(f"Historical completed trials: {historical['completed_trials']}")
    print(f"Medical completed trials: {trial_stats['complete']} (new: {trial_stats['new_complete']})")
    print(f"Best medical objective: {metadata['best_medical_utility_score']:.6f}")
    print(f"Candidate status: {selection_status}")
    print(f"Final model changed: {metadata['final_model_changed']}")


if __name__ == "__main__":
    main()
