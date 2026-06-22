from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


ID_TOKENS = ("variant_id", "identifier", "patient_id", "sample_id", "row_index")
TARGET_ENCODING_TOKENS = ("_te", "target_encoding")


@dataclass(frozen=True)
class FeatureSelectionResult:
    feature_sets: dict[str, list[str]]
    audit: pd.DataFrame
    stable_ranking: pd.DataFrame
    leakage_suspicion: pd.DataFrame


def numeric_model_features(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in {"Label", "Variant_ID"} and pd.api.types.is_numeric_dtype(df[column])
    ]


def _is_near_constant(series: pd.Series, threshold: float = 0.995) -> bool:
    values = series.dropna()
    if values.empty:
        return True
    return bool(values.value_counts(normalize=True, dropna=False).iloc[0] >= threshold)


def _id_like(series: pd.Series, name: str) -> bool:
    name_lower = name.lower()
    if any(token in name_lower for token in ID_TOKENS):
        return True
    values = series.dropna()
    return bool(len(values) >= 20 and values.nunique() / len(values) > 0.985)


def _suspicious_separation(series: pd.Series, y: pd.Series) -> tuple[bool, float]:
    values = pd.to_numeric(series, errors="coerce")
    usable = values.notna() & y.notna()
    if usable.sum() < 20 or y[usable].nunique() < 2:
        return False, np.nan
    benign = values[usable & y.eq(0)]
    pathogenic = values[usable & y.eq(1)]
    if benign.empty or pathogenic.empty:
        return False, np.nan
    exact_separation = benign.max() < pathogenic.min() or pathogenic.max() < benign.min()
    corr = float(pd.Series(values[usable]).corr(y[usable], method="spearman"))
    suspicious = bool(exact_separation or abs(corr) >= 0.995)
    return suspicious, corr


def _duplicate_columns(df: pd.DataFrame, columns: Iterable[str]) -> set[str]:
    seen: dict[tuple[float, ...], str] = {}
    duplicates: set[str] = set()
    for column in columns:
        values = tuple(pd.to_numeric(df[column], errors="coerce").fillna(-987654.321).round(12).to_numpy())
        if values in seen:
            duplicates.add(column)
        else:
            seen[values] = column
    return duplicates


def build_feature_selection(
    train_df: pd.DataFrame,
    y: pd.Series | np.ndarray | None = None,
    correlation_threshold: float = 0.985,
    compact_max_features: int = 180,
) -> FeatureSelectionResult:
    """Fit safety-oriented selection using training rows only.

    The output intentionally treats suspicious features as exclusions even when
    they improve training discrimination.  It is safe to call independently in
    every CV fold.
    """
    labels = pd.Series(y, index=train_df.index) if y is not None else train_df.get("Label")
    candidates = numeric_model_features(train_df)
    duplicate = _duplicate_columns(train_df, candidates)
    audit_rows: list[dict[str, object]] = []
    keep_stage1: list[str] = []
    for column in candidates:
        series = train_df[column]
        near_constant = _is_near_constant(series)
        id_like = _id_like(series, column)
        reason = "keep"
        if column in duplicate:
            reason = "duplicate"
        elif near_constant:
            reason = "constant_or_near_constant"
        elif id_like:
            reason = "id_like"
        audit_rows.append(
            {
                "feature": column,
                "stage1_reason": reason,
                "missing_rate": float(series.isna().mean()),
                "unique_ratio": float(series.nunique(dropna=True) / max(series.notna().sum(), 1)),
            }
        )
        if reason == "keep":
            keep_stage1.append(column)

    leakage_rows: list[dict[str, object]] = []
    keep_stage2: list[str] = []
    for column in keep_stage1:
        suspicious, correlation = _suspicious_separation(train_df[column], labels) if labels is not None else (False, np.nan)
        leakage_rows.append(
            {
                "feature": column,
                "suspicious": suspicious,
                "spearman_target_correlation": correlation,
                "reason": "near_perfect_separation_or_correlation" if suspicious else "none",
            }
        )
        if not suspicious:
            keep_stage2.append(column)

    representatives: list[str] = []
    if keep_stage2:
        corr = train_df[keep_stage2].corr(method="spearman", min_periods=20).abs()
        seen: set[str] = set()
        for column in keep_stage2:
            if column in seen:
                continue
            cluster = set(corr.index[corr[column].ge(correlation_threshold)])
            # Prefer lower missingness and non-target-encoded representatives.
            representative = min(
                cluster,
                key=lambda name: (
                    float(train_df[name].isna().mean()),
                    int(any(token in name.lower() for token in TARGET_ENCODING_TOKENS)),
                    name,
                ),
            )
            representatives.append(representative)
            seen.update(cluster)

    ranking = pd.DataFrame({"feature": representatives})
    if labels is not None and len(representatives) and labels.nunique() == 2:
        matrix = train_df[representatives].replace([np.inf, -np.inf], np.nan)
        matrix = matrix.fillna(matrix.median()).fillna(0.0)
        try:
            importance = mutual_info_classif(matrix, labels.astype(int), random_state=42)
        except ValueError:
            importance = np.zeros(len(representatives), dtype=float)
        ranking["mutual_information"] = importance
    else:
        ranking["mutual_information"] = 0.0
    ranking["missing_rate"] = ranking["feature"].map(lambda col: float(train_df[col].isna().mean()))
    ranking = ranking.sort_values(["mutual_information", "missing_rate", "feature"], ascending=[False, True, True]).reset_index(drop=True)
    ranking["stability_rank"] = np.arange(1, len(ranking) + 1)

    full_safe = representatives
    compact = ranking.head(compact_max_features)["feature"].tolist()
    no_population = [c for c in full_safe if not any(token in c.lower() for token in ("af", "pop", "cat1", "ba1", "bs1", "pm2", "bs2"))]
    no_target_encoded = [c for c in full_safe if not any(token in c.lower() for token in TARGET_ENCODING_TOKENS)]
    no_high_suspicion = full_safe.copy()
    bio_advanced = [c for c in full_safe if c.startswith("adv_")]
    ensemble = compact if compact else full_safe
    result = FeatureSelectionResult(
        feature_sets={
            "full_safe_features": full_safe,
            "compact_stable_features": compact,
            "no_population_features": no_population,
            "no_target_encoded_features": no_target_encoded,
            "no_high_suspicion_features": no_high_suspicion,
            "bio_advanced_features": bio_advanced,
            "ensemble_feature_set": ensemble,
        },
        audit=pd.DataFrame(audit_rows),
        stable_ranking=ranking,
        leakage_suspicion=pd.DataFrame(leakage_rows),
    )
    return result


def save_selection_reports(result: FeatureSelectionResult, tables_dir: str | Path) -> None:
    target = Path(tables_dir)
    target.mkdir(parents=True, exist_ok=True)
    feature_rows = [
        {"feature_set": name, "feature": feature, "n_features": len(features)}
        for name, features in result.feature_sets.items()
        for feature in features
    ]
    pd.DataFrame(feature_rows).to_csv(target / "advanced_feature_list.csv", index=False)
    result.audit.to_csv(target / "feature_generation_audit.csv", index=False)
    result.stable_ranking.to_csv(target / "stable_feature_ranking.csv", index=False)
    result.leakage_suspicion.to_csv(target / "leakage_suspicion_report.csv", index=False)


def _feature_group(feature: str) -> str:
    name = feature.lower()
    if name.startswith("adv_"):
        return "advanced_bio"
    if any(token in name for token in TARGET_ENCODING_TOKENS):
        return "target_encoded"
    if any(token in name for token in ("af", "pop", "cat1", "ba1", "bs1", "pm2", "bs2")):
        return "population_frequency"
    if name.startswith("ek_") or "ek_" in name:
        return "computational_evidence"
    if name.startswith("al_") or "al_" in name:
        return "annotation_numeric"
    if name.startswith("miss_"):
        return "missingness"
    if name.startswith("aa_") or "blosum" in name or "grantham" in name:
        return "amino_acid"
    return "other"


def run_feature_stability_gate(prepared, reports_dir: str | Path = "reports") -> dict[str, pd.DataFrame]:
    """Evaluate feature sets and fold stability using train-fold-only selection.

    This is intentionally a separate expensive stage. It trains a conservative
    LightGBM within each fold for every candidate feature set, then records OOF
    ablation outcomes and permutation/gain/SHAP stability for the full-safe set.
    """
    from pathlib import Path

    from advanced_bio_features import AdvancedBioFeatureEngineer
    from final_model_zoo import _model_matrix, _select_columns, default_model_specs
    from final_thresholding import select_threshold_candidates, threshold_grid
    from medical_metrics import compute_medical_metrics
    from teknofest.features import FeatureEngineer, detect_binary_al_cols
    from teknofest.training import make_lgbm
    from teknofest.validation import contamination_aware_folds

    reports = Path(reports_dir)
    tables = reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    spec = next(item for item in default_model_specs() if item.model_id == "lightgbm_conservative_regularized")
    folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask, n_splits=5, seed=42)
    set_predictions: dict[str, list[pd.DataFrame]] = {}
    importance_rows: list[dict[str, object]] = []
    audit_rows: list[pd.DataFrame] = []
    leakage_rows: list[pd.DataFrame] = []

    for fold in folds:
        flags = detect_binary_al_cols(prepared.master, prepared.al_cols)
        engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flags)
        advanced = AdvancedBioFeatureEngineer()
        train = advanced.transform(engineer.fit_transform(prepared.master.iloc[fold.train_idx].copy()))
        validation = advanced.transform(engineer.transform(prepared.master.iloc[fold.val_idx].copy()))
        selection = build_feature_selection(train, train["Label"])
        fold_audit = selection.audit.copy()
        fold_audit["fold"] = fold.fold
        audit_rows.append(fold_audit)
        fold_leakage = selection.leakage_suspicion.copy()
        fold_leakage["fold"] = fold.fold
        leakage_rows.append(fold_leakage)
        for feature_set, columns in selection.feature_sets.items():
            if not columns:
                continue
            model = make_lgbm(spec.params)
            x_train = _model_matrix(train, columns)
            x_val = _model_matrix(validation, columns)
            model.fit(x_train, train["Label"].astype(int))
            probability = model.predict_proba(x_val)[:, 1]
            set_predictions.setdefault(feature_set, []).append(
                pd.DataFrame({"fold": fold.fold, "Label": validation["Label"].astype(int), "probability": probability})
            )
            if feature_set != "full_safe_features":
                continue
            gains = pd.Series(model.feature_importances_, index=columns, dtype=float)
            for feature, value in gains.items():
                importance_rows.append({"feature": feature, "fold": fold.fold, "importance_type": "gain", "importance": float(value)})
            try:
                from sklearn.inspection import permutation_importance

                sampled = validation.sample(min(len(validation), 250), random_state=fold.fold)
                permutation = permutation_importance(
                    model,
                    _model_matrix(sampled, columns),
                    sampled["Label"].astype(int),
                    scoring="average_precision",
                    n_repeats=3,
                    random_state=42,
                    n_jobs=-1,
                )
                for feature, value in zip(columns, permutation.importances_mean):
                    importance_rows.append({"feature": feature, "fold": fold.fold, "importance_type": "permutation", "importance": float(value)})
            except (ImportError, ValueError):
                pass
            try:
                import shap

                sampled = validation.sample(min(len(validation), 250), random_state=fold.fold)
                values = shap.TreeExplainer(model).shap_values(_model_matrix(sampled, columns))
                if isinstance(values, list):
                    values = values[-1]
                for feature, value in zip(columns, np.abs(np.asarray(values)).mean(axis=0)):
                    importance_rows.append({"feature": feature, "fold": fold.fold, "importance_type": "shap", "importance": float(value)})
            except (ImportError, ValueError, AttributeError):
                pass

    comparison_rows: list[dict[str, object]] = []
    for feature_set, frames in set_predictions.items():
        oof = pd.concat(frames, ignore_index=True)
        candidates = select_threshold_candidates(threshold_grid(oof["Label"], oof["probability"]))
        choice = candidates[candidates["threshold_strategy"].eq("max_medical_utility")].iloc[0]
        metrics = compute_medical_metrics(oof["Label"], oof["probability"], float(choice["threshold"]))
        metrics.update({"feature_set": feature_set, "threshold_strategy": "max_medical_utility", "n_oof": len(oof)})
        comparison_rows.append(metrics)
    comparison = pd.DataFrame(comparison_rows).sort_values("medical_utility_score", ascending=False)
    comparison.to_csv(tables / "feature_set_comparison.csv", index=False)
    comparison.to_csv(tables / "feature_group_ablation_final.csv", index=False)
    audit = pd.concat(audit_rows, ignore_index=True)
    audit.to_csv(tables / "feature_generation_audit.csv", index=False)
    pd.concat(leakage_rows, ignore_index=True).to_csv(tables / "leakage_suspicion_report.csv", index=False)

    raw_importance = pd.DataFrame(importance_rows)
    if raw_importance.empty:
        ranking = pd.DataFrame(columns=["feature", "feature_group", "permutation_importance_mean", "shap_importance_mean", "gain_importance_mean", "fold_presence"])
    else:
        ranking = raw_importance.pivot_table(index="feature", columns="importance_type", values="importance", aggfunc="mean").reset_index()
        for column in ("permutation", "shap", "gain"):
            if column not in ranking.columns:
                ranking[column] = np.nan
        presence = raw_importance.groupby("feature")["fold"].nunique().rename("fold_presence")
        ranking = ranking.merge(presence, on="feature", how="left")
        ranking["feature_group"] = ranking["feature"].map(_feature_group)
        ranking = ranking.rename(
            columns={
                "permutation": "permutation_importance_mean",
                "shap": "shap_importance_mean",
                "gain": "gain_importance_mean",
            }
        )
        ranking["stability_score"] = ranking["fold_presence"].fillna(0) / max(len(folds), 1)
        ranking = ranking.sort_values(
            ["stability_score", "permutation_importance_mean", "shap_importance_mean", "gain_importance_mean"],
            ascending=[False, False, False, False],
        )
    ranking.to_csv(tables / "stable_feature_ranking.csv", index=False)
    if ranking.empty:
        group_importance = pd.DataFrame(columns=["feature_group", "n_features", "permutation_importance_mean", "shap_importance_mean", "gain_importance_mean", "stability_score"])
    else:
        group_importance = ranking.groupby("feature_group", as_index=False).agg(
            n_features=("feature", "size"),
            permutation_importance_mean=("permutation_importance_mean", "mean"),
            shap_importance_mean=("shap_importance_mean", "mean"),
            gain_importance_mean=("gain_importance_mean", "mean"),
            stability_score=("stability_score", "mean"),
        )
    group_importance.to_csv(tables / "acmg_feature_group_importance.csv", index=False)
    best = comparison.iloc[0] if not comparison.empty else pd.Series(dtype=object)
    text = [
        "# Feature Selection Final Decision",
        "",
        "Constant, duplicate, ID-like, and near-perfect-separation features are removed inside each training fold. Correlated clusters retain the lower-missingness representative. Feature-set decisions are based on conservative LightGBM OOF MedicalUtilityScore, not training accuracy.",
        "",
        f"Best evaluated feature set: `{best.get('feature_set', 'not available')}`.",
    ]
    (reports / "feature_selection_final_decision.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    advanced_summary = [
        "# Advanced Feature Engineering Summary",
        "",
        "Advanced amino-acid substitution, ACMG-inspired evidence proxy, AL/EK aggregation, and interaction features are deterministic and label-free. Target encoding remains part of the base FeatureEngineer and is fit only in training folds for OOF work.",
        "",
        f"Advanced features available after transformation: {sum(name.startswith('adv_') for name in train.columns)}.",
    ]
    (reports / "advanced_feature_engineering_summary.md").write_text("\n".join(advanced_summary) + "\n", encoding="utf-8")
    return {"comparison": comparison, "ranking": ranking, "audit": audit}
