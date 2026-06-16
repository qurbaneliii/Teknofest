from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from teknofest.data_prep import DATASET_FILES, load_datasets


def cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def auc_equivalent(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    valid = x.notna() & y.notna()
    x = pd.to_numeric(x[valid], errors="coerce")
    y = y[valid]
    valid = x.notna()
    x = x[valid]
    y = y[valid]
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan, np.nan
    res = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    auc = float(res.statistic / (len(pos) * len(neg)))
    effect = float(2 * auc - 1)
    return auc, float(res.pvalue), effect


def missingness_band(rate: float) -> str:
    pct = rate * 100
    if pct == 0:
        return "0%"
    if pct <= 25:
        return "1-25%"
    if pct <= 50:
        return "26-50%"
    if pct <= 75:
        return "51-75%"
    if pct < 100:
        return "76-99%"
    return "100%"


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def plot_missingness(master: pd.DataFrame, out_dir: Path) -> None:
    miss = master.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    miss.head(80).sort_values().plot(kind="barh", color="#425e7a")
    plt.xlabel("Missingness rate")
    plt.title("Top 80 MASTER columns by missingness")
    plt.tight_layout()
    plt.savefig(out_dir / "missingness_top80.png", dpi=170)
    plt.close()


def plot_ek_corr(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ek_cols = cols(master, "EK_")
    corr = master[ek_cols].corr(method="spearman")
    plt.figure(figsize=(8, 7))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman rho")
    plt.xticks(range(len(ek_cols)), ek_cols, rotation=45, ha="right")
    plt.yticks(range(len(ek_cols)), ek_cols)
    plt.title("EK Spearman correlation")
    plt.tight_layout()
    plt.savefig(out_dir / "ek_spearman_correlation.png", dpi=170)
    plt.close()
    return corr


def plot_top_numeric_boxplots(master: pd.DataFrame, top_features: list[str], out_dir: Path) -> None:
    selected = top_features[:10]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, feature in zip(axes.ravel(), selected):
        data = [
            pd.to_numeric(master.loc[master["Label"] == label, feature], errors="coerce").dropna()
            for label in [0, 1]
        ]
        ax.boxplot(data, labels=["Benign", "Pathogenic"], showfliers=False)
        ax.set_title(feature)
    plt.tight_layout()
    plt.savefig(out_dir / "top10_numeric_boxplots.png", dpi=170)
    plt.close()


def plot_al_pca(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    al_cols = cols(master, "AL_")
    x = master[al_cols].apply(pd.to_numeric, errors="coerce")
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x)
    n_components = min(20, len(al_cols), len(master))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(x)
    pca_df = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, n_components + 1)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, n_components + 1), pca_df["cumulative_explained_variance"], marker="o")
    plt.xlabel("Number of AL PCs")
    plt.ylabel("Cumulative explained variance")
    plt.title("AL block PCA")
    plt.tight_layout()
    plt.savefig(out_dir / "al_pca_cumulative_variance.png", dpi=170)
    plt.close()
    return pca_df


def run_eda(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    data = load_datasets(data_dir)
    master = data["master"]
    panel_names = ["kanser", "pah", "cftr"]

    all_cols = list(master.columns)
    schema_rows = []
    for name, df in data.items():
        schema_rows.append(
            {
                "dataset": name.upper(),
                "rows": len(df),
                "columns": len(df.columns),
                "schema_identical_to_master": list(df.columns) == all_cols,
                "duplicate_variant_id": int(df["Variant_ID"].duplicated().sum()),
                "pathogenic_rate": float(df["Label"].mean()),
                "pathogenic": int(df["Label"].sum()),
                "benign": int((df["Label"] == 0).sum()),
            }
        )
    schema_df = pd.DataFrame(schema_rows)
    schema_df.to_csv(tables_dir / "phase1_schema_class_balance.csv", index=False)

    master_ids = set(master["Variant_ID"])
    panel_ids = set().union(*(set(data[name]["Variant_ID"]) for name in panel_names))
    overlap_rows = []
    for name in panel_names:
        ids = set(data[name]["Variant_ID"])
        unique_df = data[name][~data[name]["Variant_ID"].isin(master_ids)]
        shared_df = data[name][data[name]["Variant_ID"].isin(master_ids)]
        overlap_rows.append(
            {
                "panel": name.upper(),
                "master_overlap": len(master_ids & ids),
                "panel_unique": len(unique_df),
                "panel_unique_pathogenic_rate": float(unique_df["Label"].mean()),
                "panel_shared_pathogenic_rate": float(shared_df["Label"].mean()),
            }
        )
    master_only = master[master["Variant_ID"].isin(master_ids - panel_ids)]
    master_shared = master[master["Variant_ID"].isin(master_ids & panel_ids)]
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(tables_dir / "phase1_overlaps.csv", index=False)

    constant_rows = []
    for col in master.columns:
        nonnull = master[col].dropna()
        if nonnull.empty:
            top_rate = np.nan
            top_value = None
        else:
            vc = nonnull.value_counts(dropna=False)
            top_value = vc.index[0]
            top_rate = float(vc.iloc[0] / len(nonnull))
        if nonnull.nunique(dropna=False) <= 1 or top_rate > 0.99:
            constant_rows.append(
                {
                    "column": col,
                    "n_nonnull": len(nonnull),
                    "n_unique_nonnull": nonnull.nunique(dropna=False),
                    "top_value": top_value,
                    "top_value_rate_nonnull": top_rate,
                }
            )
    constants_df = pd.DataFrame(constant_rows)
    constants_df.to_csv(tables_dir / "phase1_constant_near_constant.csv", index=False)

    missing_df = pd.DataFrame(
        {
            "column": master.columns,
            "missing_rate": master.isna().mean().to_numpy(),
            "missing_count": master.isna().sum().to_numpy(),
        }
    )
    missing_df["band"] = missing_df["missing_rate"].map(missingness_band)
    missing_df.to_csv(tables_dir / "phase1_missingness_profile.csv", index=False)
    missing_df["band"].value_counts().rename_axis("band").reset_index(name="columns").to_csv(
        tables_dir / "phase1_missingness_bands.csv", index=False
    )
    plot_missingness(master, figures_dir)

    al_cols = cols(master, "AL_")
    cat_cols = cols(master, "CAT_")
    ek_cols = cols(master, "EK_")
    aa_cols = ["AA_1", "AA_2"]
    numeric_cols = al_cols + ek_cols

    al_summary = master[al_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    al_summary["missing_rate"] = master[al_cols].isna().mean()
    al_summary["skew"] = master[al_cols].skew(numeric_only=True)
    al_summary["n_notna"] = master[al_cols].notna().sum()
    al_summary["is_binary_01"] = [
        set(pd.to_numeric(master[c], errors="coerce").dropna().unique()).issubset({0, 1, 0.0, 1.0})
        and master[c].notna().any()
        for c in al_cols
    ]
    al_summary.to_csv(tables_dir / "phase2_al_distribution_summary.csv")
    al_summary.groupby(["n_notna", "is_binary_01"]).size().reset_index(name="columns").to_csv(
        tables_dir / "phase2_al_groups_by_n_notna.csv", index=False
    )

    cat_rows = []
    cat_label_rows = []
    base_rate = master["Label"].mean()
    for col in cat_cols:
        value_counts = master[col].fillna("missing").value_counts()
        for value, count in value_counts.items():
            subset = master[master[col].fillna("missing") == value]
            label_rate = float(subset["Label"].mean())
            cat_label_rows.append(
                {
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "label_rate": label_rate,
                    "delta_from_base_rate": label_rate - base_rate,
                    "deviates_gt_20pp": abs(label_rate - base_rate) > 0.20,
                }
            )
        cat_text = master[col].astype("string")
        cat_rows.append(
            {
                "column": col,
                "unique_values": int(master[col].nunique(dropna=True)),
                "missing_rate": float(master[col].isna().mean()),
                "composite_rate": float(cat_text.str.contains(r"&|,|\|", na=False).mean()),
            }
        )
    cat_df = pd.DataFrame(cat_rows)
    cat_label_df = pd.DataFrame(cat_label_rows).sort_values(["column", "label_rate"])
    cat_df.to_csv(tables_dir / "phase2_cat_summary.csv", index=False)
    cat_label_df.to_csv(tables_dir / "phase2_cat_label_rates.csv", index=False)

    ek_summary = master[ek_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    ek_summary["missing_rate"] = master[ek_cols].isna().mean()
    ek_summary["has_negative"] = (master[ek_cols] < 0).any()
    ek_summary.to_csv(tables_dir / "phase2_ek_summary.csv")
    ek_corr = plot_ek_corr(master, figures_dir)
    ek_corr.to_csv(tables_dir / "phase2_ek_spearman_correlation.csv")

    aa_rows = []
    for col in aa_cols:
        for value, count in master[col].fillna("missing").value_counts().items():
            aa_rows.append({"column": col, "value": value, "count": int(count)})
    pd.DataFrame(aa_rows).to_csv(tables_dir / "phase2_aa_frequency.csv", index=False)
    aa_pair_series = master["AA_1"].astype(str) + ">" + master["AA_2"].astype(str)
    aa_pairs = aa_pair_series.value_counts().reset_index()
    aa_pairs.columns = ["aa_pair", "count"]
    aa_pairs.to_csv(tables_dir / "phase2_aa_pairs.csv", index=False)
    standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
    nonstandard = sorted(
        set(master["AA_1"].dropna().astype(str)) | set(master["AA_2"].dropna().astype(str)) - standard_aas
    )
    pd.DataFrame({"nonstandard_or_unexpected_code": nonstandard}).to_csv(
        tables_dir / "phase2_aa_nonstandard_codes.csv", index=False
    )

    aggregate = pd.DataFrame(index=master.index)
    aggregate["n_pops"] = master[[f"AL_{i}" for i in range(1, 27)]].notna().sum(axis=1)
    aggregate["n_nonmiss_AL"] = master[al_cols].notna().sum(axis=1)
    aggregate["max_AF"] = master[[f"AL_{i}" for i in range(1, 27)]].max(axis=1).fillna(0.0)
    aggregate["min_AF_nz"] = master[[f"AL_{i}" for i in range(1, 27)]].replace(0, np.nan).min(axis=1)
    aggregate["EK7xEK9"] = master["EK_7"] * master["EK_9"]
    aggregate["EK7_minus_EK9"] = master["EK_7"] - master["EK_9"]
    numeric_for_auc = pd.concat([master[numeric_cols], aggregate], axis=1)

    auc_rows = []
    for col in numeric_for_auc.columns:
        auc, pvalue, effect = auc_equivalent(numeric_for_auc[col], master["Label"])
        auc_rows.append(
            {
                "feature": col,
                "auc_equivalent": auc,
                "auc_directional_strength": abs(auc - 0.5) + 0.5 if pd.notna(auc) else np.nan,
                "p_value": pvalue,
                "rank_biserial": effect,
                "n_notna": int(numeric_for_auc[col].notna().sum()),
            }
        )
    auc_df = pd.DataFrame(auc_rows).sort_values("auc_directional_strength", ascending=False)
    auc_df.to_csv(tables_dir / "phase3_numeric_auc_ranked.csv", index=False)
    plot_top_numeric_boxplots(master.join(aggregate), auc_df["feature"].head(10).tolist(), figures_dir)

    miss_signal_rows = []
    for col in master.columns:
        rate = master[col].isna().mean()
        if 0 < rate < 1:
            miss = master[col].isna().astype(float)
            corr = miss.corr(master["Label"])
            miss_signal_rows.append(
                {
                    "column": col,
                    "missing_rate": rate,
                    "label_rate_when_missing": float(master.loc[master[col].isna(), "Label"].mean()),
                    "label_rate_when_present": float(master.loc[master[col].notna(), "Label"].mean()),
                    "label_rate_delta_missing_minus_present": float(
                        master.loc[master[col].isna(), "Label"].mean()
                        - master.loc[master[col].notna(), "Label"].mean()
                    ),
                    "missing_label_corr": float(corr),
                }
            )
    miss_signal_df = pd.DataFrame(miss_signal_rows).sort_values(
        "missing_label_corr", key=lambda s: s.abs(), ascending=False
    )
    miss_signal_df.to_csv(tables_dir / "phase3_missingness_as_feature.csv", index=False)

    triplet_rows = []
    for start in range(27, 334, 3):
        triple = [f"AL_{i}" for i in range(start, min(start + 3, 335)) if f"AL_{i}" in master.columns]
        if len(triple) == 3:
            corr = master[triple].corr(method="spearman")
            vals = [corr.iloc[0, 1], corr.iloc[0, 2], corr.iloc[1, 2]]
            mean_abs_rho = np.nan if pd.isna(vals).all() else float(np.nanmean(np.abs(vals)))
            triplet_rows.append(
                {
                    "columns": ",".join(triple),
                    "rho_12": vals[0],
                    "rho_13": vals[1],
                    "rho_23": vals[2],
                    "mean_abs_rho": mean_abs_rho,
                }
            )
    triplet_df = pd.DataFrame(triplet_rows)
    triplet_df.to_csv(tables_dir / "phase4_al_triplet_correlations.csv", index=False)

    cross = master[ek_cols].join(aggregate).corr(method="spearman")
    cross.to_csv(tables_dir / "phase4_cross_group_spearman.csv")
    high_cross = []
    for c1 in ek_cols:
        for c2 in aggregate.columns:
            rho = cross.loc[c1, c2]
            if pd.notna(rho) and abs(rho) > 0.5:
                high_cross.append({"feature_1": c1, "feature_2": c2, "spearman_rho": rho})
    high_cross_df = pd.DataFrame(high_cross)
    high_cross_df.to_csv(tables_dir / "phase4_high_cross_group_correlations.csv", index=False)
    pca_df = plot_al_pca(master, figures_dir)
    pca_df.to_csv(tables_dir / "phase4_al_pca.csv", index=False)
    leakage_df = auc_df[auc_df["auc_directional_strength"] > 0.85]
    leakage_df.to_csv(tables_dir / "phase4_leakage_suspects.csv", index=False)

    top10 = auc_df["feature"].head(10).tolist()
    shift_rows = []
    for name, df in data.items():
        agg = pd.DataFrame(index=df.index)
        agg["n_pops"] = df[[f"AL_{i}" for i in range(1, 27)]].notna().sum(axis=1)
        agg["n_nonmiss_AL"] = df[al_cols].notna().sum(axis=1)
        agg["max_AF"] = df[[f"AL_{i}" for i in range(1, 27)]].max(axis=1).fillna(0.0)
        agg["min_AF_nz"] = df[[f"AL_{i}" for i in range(1, 27)]].replace(0, np.nan).min(axis=1)
        agg["EK7xEK9"] = df["EK_7"] * df["EK_9"]
        agg["EK7_minus_EK9"] = df["EK_7"] - df["EK_9"]
        merged = df.join(agg)
        for feature in top10:
            if feature not in merged.columns:
                continue
            shift_rows.append(
                {
                    "dataset": name.upper(),
                    "feature": feature,
                    "mean": float(pd.to_numeric(merged[feature], errors="coerce").mean()),
                    "median": float(pd.to_numeric(merged[feature], errors="coerce").median()),
                    "pathogenic_mean": float(pd.to_numeric(merged.loc[merged["Label"] == 1, feature], errors="coerce").mean()),
                    "benign_mean": float(pd.to_numeric(merged.loc[merged["Label"] == 0, feature], errors="coerce").mean()),
                    "missing_rate": float(merged[feature].isna().mean()),
                }
            )
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(tables_dir / "phase5_top10_cross_dataset_shift.csv", index=False)

    panel_unique_rows = []
    for name in panel_names:
        df = data[name]
        unique = df[~df["Variant_ID"].isin(master_ids)]
        shared = df[df["Variant_ID"].isin(master_ids)]
        for subset_name, subset in [("unique", unique), ("shared_with_master", shared)]:
            panel_unique_rows.append(
                {
                    "panel": name.upper(),
                    "subset": subset_name,
                    "n": len(subset),
                    "pathogenic_rate": float(subset["Label"].mean()),
                    "overall_missing_rate": float(subset.isna().mean().mean()),
                    "numeric_range_mean": float(
                        subset[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min()).mean()
                    ),
                }
            )
    panel_unique_df = pd.DataFrame(panel_unique_rows)
    panel_unique_df.to_csv(tables_dir / "phase5_panel_unique_vs_shared.csv", index=False)

    acmg_rows = []
    for _, row in auc_df.head(25).iterrows():
        f = row["feature"]
        if f in ["max_AF", "AL_26"] or f.startswith("AL_"):
            criterion = "BA1/BS1/PM2"
            meaning = "Population frequency or rarity evidence; direction must be interpreted from AUC and distributions."
            confidence = "medium"
        elif f in ["n_pops", "n_nonmiss_AL"]:
            criterion = "PM2/BS2 proxy"
            meaning = "Database absence/presence aggregate; missingness encodes rarity and ascertainment."
            confidence = "high"
        elif f.startswith("EK_") or f.startswith("EK7"):
            criterion = "PP3/BP4"
            meaning = "Computational/evolutionary pathogenicity evidence."
            confidence = "medium"
        elif f.startswith("AA") or f.startswith("aa"):
            criterion = "PM5/PP3 proxy"
            meaning = "Amino-acid substitution chemistry or learned substitution prior."
            confidence = "low"
        else:
            criterion = "technical/statistical artifact candidate"
            meaning = "Predictive behavior does not cleanly map to a single ACMG criterion."
            confidence = "low"
        acmg_rows.append(
            {
                "feature": f,
                "auc_equivalent": row["auc_equivalent"],
                "proposed_acmg_category": criterion,
                "justification": meaning,
                "confidence": confidence,
            }
        )
    acmg_df = pd.DataFrame(acmg_rows)
    acmg_df.to_csv(tables_dir / "phase6_acmg_interpretation.csv", index=False)

    report = build_report(
        schema_df,
        overlap_df,
        master_only,
        master_shared,
        constants_df,
        missing_df,
        al_summary,
        cat_df,
        cat_label_df,
        ek_summary,
        ek_corr,
        aa_pairs,
        auc_df,
        miss_signal_df,
        triplet_df,
        high_cross_df,
        pca_df,
        leakage_df,
        shift_df,
        panel_unique_df,
        acmg_df,
        figures_dir,
    )
    (out_dir / "EDA_REPORT.md").write_text(report, encoding="utf-8")


def build_report(
    schema_df,
    overlap_df,
    master_only,
    master_shared,
    constants_df,
    missing_df,
    al_summary,
    cat_df,
    cat_label_df,
    ek_summary,
    ek_corr,
    aa_pairs,
    auc_df,
    miss_signal_df,
    triplet_df,
    high_cross_df,
    pca_df,
    leakage_df,
    shift_df,
    panel_unique_df,
    acmg_df,
    figures_dir,
) -> str:
    top10 = auc_df.head(10)[["feature", "auc_equivalent", "auc_directional_strength", "p_value", "n_notna"]]
    strongest_missing = miss_signal_df.head(10)
    al_group_count = al_summary.groupby("n_notna").size().sort_values(ascending=False).head(10).reset_index(name="columns")
    leakage_text = "None above 0.85." if leakage_df.empty else markdown_table(leakage_df[["feature", "auc_equivalent", "auc_directional_strength"]], 20)
    ek_high = []
    for c1 in ek_corr.columns:
        for c2 in ek_corr.columns:
            if c1 < c2 and abs(ek_corr.loc[c1, c2]) > 0.5:
                ek_high.append({"feature_1": c1, "feature_2": c2, "spearman_rho": ek_corr.loc[c1, c2]})
    ek_high_df = pd.DataFrame(ek_high).sort_values("spearman_rho", ascending=False) if ek_high else pd.DataFrame()

    return f"""# TEKNOFEST 2026 EDA Report

## Executive Summary

The 10 most predictive individual numeric features on MASTER by Mann-Whitney AUC-equivalent are:

{markdown_table(top10, 10)}

The single most important AL structural discovery is that AL columns cluster strongly by identical non-null counts. The largest discovered availability groups are:

{markdown_table(al_group_count, 10)}

The most important missingness finding is that missingness itself is label-associated for many columns. The strongest missingness signals are:

{markdown_table(strongest_missing[['column','missing_rate','label_rate_when_missing','label_rate_when_present','missing_label_corr']], 10)}

The most important cross-dataset finding is that panel-unique variants have very different pathogenic rates from panel variants shared with MASTER:

{markdown_table(panel_unique_df, 10)}

Leakage suspects with AUC-equivalent directional strength above 0.85:

{leakage_text}

## Phase 1 - Data Integrity And Structure

All four files were loaded from the raw competition folder and compared against MASTER column order.

{markdown_table(schema_df, 10)}

Overlap counts and panel-unique subsets:

{markdown_table(overlap_df, 10)}

MASTER-only has n={len(master_only)} and pathogenic rate {master_only['Label'].mean():.4f}. MASTER-shared-with-panel has n={len(master_shared)} and pathogenic rate {master_shared['Label'].mean():.4f}.

Constant or near-constant columns (>99% same non-null value):

{markdown_table(constants_df, 25)}

Missingness bands:

{markdown_table(missing_df['band'].value_counts().rename_axis('band').reset_index(name='columns'), 10)}

Figure: `{figures_dir / 'missingness_top80.png'}`

## Phase 2 - Univariate Analysis By Feature Group

### AL Columns

AL columns were profiled for distribution, missingness, binary status, and grouped by identical non-null counts. There are {int(al_summary['is_binary_01'].sum())} strictly binary AL columns in MASTER.

Top AL availability groups:

{markdown_table(al_group_count, 20)}

### CAT Columns

{markdown_table(cat_df, 10)}

Most label-skewed CAT categories:

{markdown_table(cat_label_df.reindex(cat_label_df['delta_from_base_rate'].abs().sort_values(ascending=False).index)[['column','value','count','label_rate','delta_from_base_rate']].head(20), 20)}

### EK Columns

{markdown_table(ek_summary.reset_index().rename(columns={'index':'feature'}), 12)}

EK feature pairs with |Spearman rho| > 0.5:

{markdown_table(ek_high_df, 20)}

Figure: `{figures_dir / 'ek_spearman_correlation.png'}`

### AA Columns

Most common amino-acid substitutions:

{markdown_table(aa_pairs, 20)}

## Phase 3 - Bivariate Analysis: Features vs Label

Top 25 numeric features:

{markdown_table(auc_df[['feature','auc_equivalent','auc_directional_strength','p_value','rank_biserial','n_notna']], 25)}

Bottom 25 numeric features:

{markdown_table(auc_df.sort_values('auc_directional_strength')[['feature','auc_equivalent','auc_directional_strength','p_value','rank_biserial','n_notna']], 25)}

Missingness-as-feature ranking:

{markdown_table(miss_signal_df[['column','missing_rate','label_rate_when_missing','label_rate_when_present','missing_label_corr']], 25)}

Figure: `{figures_dir / 'top10_numeric_boxplots.png'}`

## Phase 4 - Multivariate And Structural Analysis

Adjacent AL triplet Spearman correlations show whether apparent triplets are redundant or independent. Lowest mean absolute triplet correlations:

{markdown_table(triplet_df.sort_values('mean_abs_rho').head(15), 15)}

Highest mean absolute triplet correlations:

{markdown_table(triplet_df.sort_values('mean_abs_rho', ascending=False).head(15), 15)}

High cross-group correlations between EK columns and engineered AL aggregates:

{markdown_table(high_cross_df, 20)}

AL PCA top components:

{markdown_table(pca_df, 20)}

Figure: `{figures_dir / 'al_pca_cumulative_variance.png'}`

Leakage scan:

{leakage_text}

## Phase 5 - Cross-Dataset Comparison

Top-feature distribution shifts by dataset:

{markdown_table(shift_df, 40)}

Panel-unique versus shared variants:

{markdown_table(panel_unique_df, 10)}

KANSER differs from MASTER primarily through a much lower panel-unique pathogenic rate and a cancer-panel-specific distribution of computational and population-frequency features. PAH and CFTR panel-unique subsets are smaller and have intermediate pathogenic rates, so their estimates are less stable.

## Phase 6 - ACMG Evidence Interpretation

Strong features mapped to plausible ACMG evidence categories. These mappings are inferred from statistics and biological plausibility, not confirmed metadata.

{markdown_table(acmg_df, 25)}

Features that do not cleanly map to ACMG are marked as technical/statistical artifact candidates, especially PCA or availability-derived signals whose biology is indirect.
"""
