from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from medical_metrics import compute_medical_metrics
from teknofest.data_prep import PreparedData
from teknofest.features import FeatureEngineer, detect_binary_al_cols
from teknofest.validation import (
    best_f1_macro_threshold,
    contamination_aware_folds,
    youden_j_threshold,
)


DROP_FROM_MODEL = {
    "Variant_ID",
    "Label",
    "CAT_1",
    "CAT_2",
    "CAT_3",
    "CAT_4",
    "CAT_5",
    "CAT_6",
    "AA_1",
    "AA_2",
    "aa1_class",
    "aa2_class",
    "aa_class_change",
}


@dataclass(frozen=True)
class FoldResult:
    model: str
    fold: int
    threshold_name: str
    threshold: float
    auc_roc: float
    auc_pr: float
    f1_macro: float
    f1_weighted: float
    mcc: float
    cohen_kappa: float
    benign_precision: float
    benign_recall: float
    benign_f1: float
    pathogenic_precision: float
    pathogenic_recall: float
    pathogenic_f1: float


def model_columns(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in df.columns if c not in DROP_FROM_MODEL]
    return [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]


def align_numeric(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = model_columns(train_df)
    return train_df[cols], val_df.reindex(columns=cols)


def acmg_rule_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    score = score.mask(df["max_AF"] > 0.05, score - 8.0)
    score = score.mask((df["max_AF"] <= 0.05) & (df["max_AF"] > 0.01), score - 4.0)
    score = score + (df["n_pops"] == 0).astype(float) * 2.0
    score = score + (df["EK_7"] > 5.0).astype(float)
    score = score - (df["EK_7"] < 1.0).astype(float)
    score = score + (df["EK_9"] > 7.0).astype(float)
    score = score - (df["EK_9"] < 3.0).astype(float)
    return score


def acmg_rule_probability(df: pd.DataFrame) -> pd.Series:
    score = acmg_rule_score(df)
    return 1.0 / (1.0 + np.exp(-score))


def metric_row(
    model: str,
    fold: int,
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold_name: str,
    threshold: float,
) -> FoldResult:
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    y_pred = (y_score_arr >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return FoldResult(
        model=model,
        fold=fold,
        threshold_name=threshold_name,
        threshold=float(threshold),
        auc_roc=float(roc_auc_score(y_true_arr, y_score_arr)),
        auc_pr=float(average_precision_score(y_true_arr, y_score_arr)),
        f1_macro=float(f1_score(y_true_arr, y_pred, average="macro")),
        f1_weighted=float(f1_score(y_true_arr, y_pred, average="weighted")),
        mcc=float(matthews_corrcoef(y_true_arr, y_pred)),
        cohen_kappa=float(cohen_kappa_score(y_true_arr, y_pred)),
        benign_precision=float(precision[0]),
        benign_recall=float(recall[0]),
        benign_f1=float(f1[0]),
        pathogenic_precision=float(precision[1]),
        pathogenic_recall=float(recall[1]),
        pathogenic_f1=float(f1[1]),
    )


def threshold_metric_rows(
    model: str,
    fold: int,
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> list[FoldResult]:
    f1_threshold, _ = best_f1_macro_threshold(y_true, y_score)
    youden_threshold, _ = youden_j_threshold(y_true, y_score)
    return [
        metric_row(model, fold, y_true, y_score, "default_0.5", 0.5),
        metric_row(model, fold, y_true, y_score, "f1_macro_opt", f1_threshold),
        metric_row(model, fold, y_true, y_score, "youden_j", youden_threshold),
    ]


def fit_lr_ek() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
            ),
        ]
    )


def make_lgbm(params: dict[str, object] | None = None) -> lgb.LGBMClassifier:
    base = {
        "objective": "binary",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "scale_pos_weight": 0.35,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        base.update(params)
    return lgb.LGBMClassifier(**base)


def make_extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def fold_engineered_data(
    prepared: PreparedData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
    train_raw = prepared.master.iloc[train_idx].copy()
    val_raw = prepared.master.iloc[val_idx].copy()
    train_df = engineer.fit_transform(train_raw)
    val_df = engineer.transform(val_raw)
    return train_df, val_df


def run_cv_baselines(prepared: PreparedData, lgbm_params: dict[str, object] | None = None) -> pd.DataFrame:
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    rows: list[FoldResult] = []

    for fold in folds:
        train_df, val_df = fold_engineered_data(prepared, fold.train_idx, fold.val_idx)
        y_train = train_df["Label"]
        y_val = val_df["Label"]

        acmg_score = acmg_rule_probability(val_df)
        rows.extend(threshold_metric_rows("acmg_rule", fold.fold, y_val, acmg_score))

        ek_cols = [f"EK_{i}" for i in range(1, 10)]
        lr = fit_lr_ek()
        lr.fit(train_df[ek_cols], y_train)
        lr_score = lr.predict_proba(val_df[ek_cols])[:, 1]
        rows.extend(threshold_metric_rows("lr_ek_only", fold.fold, y_val, lr_score))

        x_train, x_val = align_numeric(train_df, val_df)
        lgbm = make_lgbm(lgbm_params)
        lgbm.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        lgbm_score = lgbm.predict_proba(x_val)[:, 1]
        rows.extend(threshold_metric_rows("lightgbm", fold.fold, y_val, lgbm_score))

        et = make_extra_trees()
        x_train_filled = x_train.fillna(-999)
        x_val_filled = x_val.fillna(-999)
        et.fit(x_train_filled, y_train)
        et_score = et.predict_proba(x_val_filled)[:, 1]
        rows.extend(threshold_metric_rows("extra_trees", fold.fold, y_val, et_score))

    return pd.DataFrame([row.__dict__ for row in rows])


def optimize_lgbm(prepared: PreparedData, n_trials: int = 100) -> tuple[dict[str, object], pd.DataFrame]:
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)

    # Reusing fold-engineered matrices keeps the Optuna loop honest and much faster.
    fold_data = []
    for fold in folds:
        train_df, val_df = fold_engineered_data(prepared, fold.train_idx, fold.val_idx)
        x_train, x_val = align_numeric(train_df, val_df)
        fold_data.append((x_train, train_df["Label"], x_val, val_df["Label"]))

    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.15, 0.55),
        }
        aucs = []
        for x_train, y_train, x_val, y_val in fold_data:
            model = make_lgbm(params)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            preds = model.predict_proba(x_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))
        mean_auc = float(np.mean(aucs))
        trial_rows.append({"trial": trial.number, "mean_auc": mean_auc, **params})
        return mean_auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return dict(study.best_params), pd.DataFrame(trial_rows)


def optimize_lgbm_resumable(
    prepared: PreparedData,
    n_trials: int = 100,
    storage_url: str | None = None,
    study_name: str = "teknofest_lgbm",
    max_estimators: int = 3000,
    timeout_seconds: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    fold_data = []
    for fold in folds:
        train_df, val_df = fold_engineered_data(prepared, fold.train_idx, fold.val_idx)
        x_train, x_val = align_numeric(train_df, val_df)
        fold_data.append((x_train, train_df["Label"], x_val, val_df["Label"]))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": max_estimators,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.15, 0.55),
        }
        aucs = []
        for x_train, y_train, x_val, y_val in fold_data:
            model = make_lgbm(params)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            preds = model.predict_proba(x_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))
        return float(np.mean(aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=True)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials = trials.rename(columns={"number": "trial", "value": "mean_auc"})
    return dict(study.best_params), trials


def _medical_fold_cache(prepared: PreparedData) -> list[dict[str, Any]]:
    """Build fold-local feature matrices once for a leakage-safe tuning run."""
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
    cache: list[dict[str, Any]] = []
    for fold in folds:
        train_df, val_df = fold_engineered_data(prepared, fold.train_idx, fold.val_idx)
        x_train, x_val = align_numeric(train_df, val_df)
        cache.append(
            {
                "fold": fold.fold,
                "train_idx": fold.train_idx,
                "val_idx": fold.val_idx,
                "x_train": x_train,
                "y_train": train_df["Label"].astype(int),
                "x_val": x_val,
                "y_val": val_df["Label"].astype(int),
            }
        )
    return cache


def _medical_search_params(trial: optuna.Trial, max_estimators: int) -> dict[str, object]:
    if max_estimators < 600:
        raise ValueError("max_estimators must be at least 600 for the controlled medical search space.")
    return {
        "n_estimators": trial.suggest_int("n_estimators", 600, min(2000, max_estimators), step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 150),
        "subsample": trial.suggest_float("subsample", 0.65, 0.95),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 15.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.30, 0.70),
    }


def _mean_metric_rows(rows: list[dict[str, float | int]]) -> dict[str, float]:
    if not rows:
        raise ValueError("At least one fold metric row is required.")
    frame = pd.DataFrame(rows)
    values: dict[str, float] = {}
    for column in (
        "roc_auc",
        "pr_auc",
        "f1_macro",
        "mcc",
        "balanced_accuracy",
        "pathogenic_recall",
        "specificity",
        "medical_utility_score",
    ):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values[f"mean_{column}"] = float(numeric.mean())
        values[f"std_{column}"] = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    return values


def medical_trials_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Expose all durable trial attributes saved in the Optuna RDB storage."""
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        row: dict[str, object] = {
            "trial": trial.number,
            "state": trial.state.name,
            "medical_utility_score": trial.value if trial.value is not None else np.nan,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update(trial.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trial").reset_index(drop=True) if rows else pd.DataFrame()


def optimize_lgbm_medical_resumable(
    prepared: PreparedData,
    n_trials: int,
    storage_url: str,
    study_name: str,
    max_estimators: int = 2000,
    timeout_seconds: int | None = None,
    resume: bool = True,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, int]]:
    """Run or resume a medical-utility Optuna study using contamination-aware folds.

    Feature engineering is fit only on each training fold. Fold thresholds are
    optimized only against their own held-out fold and are stored for stability
    review; panel labels are never read during tuning.
    """
    fold_cache = _medical_fold_cache(prepared)
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        study_name=study_name,
        load_if_exists=resume,
    )
    before = {
        "complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
    }

    def objective(trial: optuna.Trial) -> float:
        params = _medical_search_params(trial, max_estimators)
        fold_metrics: list[dict[str, float | int]] = []
        thresholds: list[float] = []
        best_iterations: list[int] = []
        for step, fold in enumerate(fold_cache):
            model = make_lgbm(params)
            model.fit(
                fold["x_train"],
                fold["y_train"],
                eval_set=[(fold["x_val"], fold["y_val"])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            probability = model.predict_proba(fold["x_val"])[:, 1]
            threshold, _ = best_f1_macro_threshold(fold["y_val"], probability)
            metrics = compute_medical_metrics(fold["y_val"], probability, threshold)
            fold_metrics.append(metrics)
            thresholds.append(float(threshold))
            best_iterations.append(int(model.best_iteration_ or params["n_estimators"]))
            current = _mean_metric_rows(fold_metrics)
            trial.report(current["mean_medical_utility_score"], step)
            trial.set_user_attr(f"fold_{int(fold['fold'])}_threshold", float(threshold))
            trial.set_user_attr(f"fold_{int(fold['fold'])}_medical_utility_score", current["mean_medical_utility_score"])
            if trial.should_prune():
                raise optuna.TrialPruned()

        summary = _mean_metric_rows(fold_metrics)
        for key, value in summary.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("threshold_mean", float(np.mean(thresholds)))
        trial.set_user_attr("threshold_median", float(np.median(thresholds)))
        trial.set_user_attr("threshold_std", float(np.std(thresholds, ddof=1)))
        trial.set_user_attr("mean_best_iteration", float(np.mean(best_iterations)))
        return summary["mean_medical_utility_score"]

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    after = {
        "complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
    }
    delta = {f"new_{key}": after[key] - before[key] for key in before}
    if not after["complete"]:
        raise RuntimeError("Medical Optuna study has no completed trials.")
    return dict(study.best_params), medical_trials_dataframe(study), {**before, **after, **delta}


def evaluate_lgbm_medical_candidate(
    prepared: PreparedData,
    params: dict[str, object],
) -> dict[str, object]:
    """Create candidate-only OOF and panel predictions; never writes a final model."""
    fold_cache = _medical_fold_cache(prepared)
    oof_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int]] = []
    gap_rows: list[dict[str, float]] = []
    thresholds: list[float] = []
    best_iterations: list[int] = []
    for fold in fold_cache:
        model = make_lgbm(params)
        model.fit(
            fold["x_train"],
            fold["y_train"],
            eval_set=[(fold["x_val"], fold["y_val"])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        probability = model.predict_proba(fold["x_val"])[:, 1]
        threshold, _ = best_f1_macro_threshold(fold["y_val"], probability)
        val_metrics = compute_medical_metrics(fold["y_val"], probability, threshold)
        train_probability = model.predict_proba(fold["x_train"])[:, 1]
        train_metrics = compute_medical_metrics(fold["y_train"], train_probability, threshold)
        fold_rows.append({"fold": int(fold["fold"]), **val_metrics})
        gap_rows.append(
            {
                "fold": int(fold["fold"]),
                "train_roc_auc": float(train_metrics["roc_auc"]),
                "validation_roc_auc": float(val_metrics["roc_auc"]),
                "roc_auc_gap": float(train_metrics["roc_auc"] - val_metrics["roc_auc"]),
            }
        )
        thresholds.append(float(threshold))
        best_iterations.append(int(model.best_iteration_ or params["n_estimators"]))
        raw = prepared.master.iloc[fold["val_idx"]]
        oof_frames.append(
            pd.DataFrame(
                {
                    "fold": int(fold["fold"]),
                    "row_index": fold["val_idx"],
                    "Variant_ID": raw["Variant_ID"].to_numpy(),
                    "Label": fold["y_val"].to_numpy(),
                    "score": probability,
                    "fold_threshold": float(threshold),
                }
            )
        )

    threshold = float(np.median(thresholds))
    oof = pd.concat(oof_frames, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof["threshold"] = threshold
    oof["prediction"] = (oof["score"] >= threshold).astype(int)
    oof_metrics = compute_medical_metrics(oof["Label"], oof["score"], threshold)

    flags = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flags)
    master = engineer.fit_transform(prepared.master.copy())
    x_master = master[model_columns(master)]
    final_params = dict(params)
    final_params["n_estimators"] = max(1, int(round(float(np.mean(best_iterations)))))
    final_model = make_lgbm(final_params)
    final_model.fit(x_master, master["Label"].astype(int))
    panel_frames: list[pd.DataFrame] = []
    for dataset, raw in (
        ("KANSER_unique", prepared.kanser_unique),
        ("PAH_unique", prepared.pah_unique),
        ("CFTR_unique", prepared.cftr_unique),
    ):
        transformed = engineer.transform(raw.copy())
        probability = final_model.predict_proba(transformed.reindex(columns=x_master.columns))[:, 1]
        panel_frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "Variant_ID": raw["Variant_ID"].to_numpy(),
                    "Label": raw["Label"].to_numpy(dtype=int),
                    "score": probability,
                }
            )
        )
    panel = pd.concat(panel_frames, ignore_index=True)
    panel["threshold"] = threshold
    panel["prediction"] = (panel["score"] >= threshold).astype(int)
    panel_metrics = compute_medical_metrics(panel["Label"], panel["score"], threshold)
    return {
        "oof_predictions": oof,
        "panel_predictions": panel,
        "fold_metrics": pd.DataFrame(fold_rows),
        "overfitting_gaps": pd.DataFrame(gap_rows),
        "oof_metrics": oof_metrics,
        "panel_metrics": panel_metrics,
        "threshold": threshold,
        "threshold_stability": float(np.std(thresholds, ddof=1)),
        "mean_roc_auc_gap": float(pd.DataFrame(gap_rows)["roc_auc_gap"].mean()),
        "mean_best_iteration": float(np.mean(best_iterations)),
        "effective_full_fit_params": final_params,
    }


def fit_final_lgbm(
    prepared: PreparedData,
    params: dict[str, object],
    model_dir: Path,
) -> tuple[FeatureEngineer, lgb.LGBMClassifier, list[str]]:
    flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
    engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
    master_df = engineer.fit_transform(prepared.master)
    x_master = master_df[model_columns(master_df)]
    y_master = master_df["Label"]

    model = make_lgbm(params)
    model.fit(x_master, y_master)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(engineer, model_dir / "feature_engineer.joblib")
    joblib.dump(model, model_dir / "lightgbm_final.joblib")
    (model_dir / "model_columns.txt").write_text("\n".join(x_master.columns) + "\n", encoding="utf-8")
    return engineer, model, list(x_master.columns)


def evaluate_panel_unique(
    prepared: PreparedData,
    engineer: FeatureEngineer,
    model: lgb.LGBMClassifier,
    columns: list[str],
) -> pd.DataFrame:
    rows: list[FoldResult] = []
    for name, raw_df in {
        "kanser_unique": prepared.kanser_unique,
        "pah_unique": prepared.pah_unique,
        "cftr_unique": prepared.cftr_unique,
    }.items():
        df = engineer.transform(raw_df)
        scores = model.predict_proba(df.reindex(columns=columns))[:, 1]
        rows.extend(threshold_metric_rows(f"lightgbm_{name}", -1, df["Label"], scores))
    return pd.DataFrame([row.__dict__ for row in rows])
