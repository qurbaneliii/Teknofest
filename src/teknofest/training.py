from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
