from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path("reports/eda/TEKNOFEST2026_EDA.ipynb")
DATA_DIR = "teknofest2026_artificialintelligenceinhealtcare-main"


def code_cell(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    nb.cells = [
        markdown_cell(
            """
# TEKNOFEST 2026 - Exploratory Data Analysis

Bu notebook `Adsiz dokuman.pdf` faylındakı EDA tələblərinə əsasən hazırlanıb.
Məqsəd yalnız verilənləri dərindən anlamaqdır: model qurulmur, hiperparametr
müzakirəsi edilmir.

Notebook mərhələləri:

1. Data integrity və struktur yoxlamaları
2. AL, CAT, EK, AA feature qruplarının univariate analizi
3. Feature vs Label bivariate analizi
4. AL triplet strukturu, cross-correlation, PCA və leakage scan
5. MASTER və panel datasetlərinin müqayisəsi
6. ACMG evidence interpretation
"""
        ),
        code_cell(
            f"""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 80)

DATA_DIR = Path("{DATA_DIR}")
OUT_DIR = Path("reports/eda_notebook_outputs")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
"""
        ),
        markdown_cell("## Helper Functions"),
        code_cell(
            """
def cols(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def auc_equivalent(x, y):
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
    rank_biserial = float(2 * auc - 1)
    directional_strength = abs(auc - 0.5) + 0.5
    return auc, float(res.pvalue), rank_biserial


def missingness_band(rate):
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
"""
        ),
        markdown_cell("## Phase 1 - Data Integrity And Structure"),
        code_cell(
            """
files = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}

data = {name: pd.read_csv(DATA_DIR / filename) for name, filename in files.items()}
master = data["MASTER"]
master_cols = list(master.columns)

schema_rows = []
for name, df in data.items():
    schema_rows.append({
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "schema_identical_to_master": list(df.columns) == master_cols,
        "duplicate_variant_id": int(df["Variant_ID"].duplicated().sum()),
        "pathogenic": int(df["Label"].sum()),
        "benign": int((df["Label"] == 0).sum()),
        "pathogenic_rate": float(df["Label"].mean()),
    })

schema_df = pd.DataFrame(schema_rows)
schema_df.to_csv(TABLE_DIR / "phase1_schema_class_balance.csv", index=False)
schema_df
"""
        ),
        code_cell(
            """
master_ids = set(data["MASTER"]["Variant_ID"])
panel_names = ["KANSER", "PAH", "CFTR"]
panel_ids = set().union(*(set(data[name]["Variant_ID"]) for name in panel_names))

overlap_rows = []
for name in panel_names:
    df = data[name]
    ids = set(df["Variant_ID"])
    unique = df[~df["Variant_ID"].isin(master_ids)]
    shared = df[df["Variant_ID"].isin(master_ids)]
    overlap_rows.append({
        "panel": name,
        "master_overlap": len(master_ids & ids),
        "panel_unique": len(unique),
        "panel_unique_pathogenic_rate": float(unique["Label"].mean()),
        "panel_shared_pathogenic_rate": float(shared["Label"].mean()),
    })

master_only = master[master["Variant_ID"].isin(master_ids - panel_ids)]
master_shared = master[master["Variant_ID"].isin(master_ids & panel_ids)]
overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_csv(TABLE_DIR / "phase1_overlaps.csv", index=False)
display(overlap_df)
print(f"MASTER-only: n={len(master_only)}, pathogenic_rate={master_only['Label'].mean():.4f}")
print(f"MASTER-shared: n={len(master_shared)}, pathogenic_rate={master_shared['Label'].mean():.4f}")
"""
        ),
        code_cell(
            """
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
        constant_rows.append({
            "column": col,
            "n_nonnull": len(nonnull),
            "n_unique_nonnull": nonnull.nunique(dropna=False),
            "top_value": top_value,
            "top_value_rate_nonnull": top_rate,
        })

constants_df = pd.DataFrame(constant_rows)
constants_df.to_csv(TABLE_DIR / "phase1_constant_near_constant.csv", index=False)
constants_df.head(30)
"""
        ),
        code_cell(
            """
missing_df = pd.DataFrame({
    "column": master.columns,
    "missing_rate": master.isna().mean().to_numpy(),
    "missing_count": master.isna().sum().to_numpy(),
})
missing_df["band"] = missing_df["missing_rate"].map(missingness_band)
missing_df.to_csv(TABLE_DIR / "phase1_missingness_profile.csv", index=False)

display(missing_df["band"].value_counts().rename_axis("band").reset_index(name="columns"))

plt.figure(figsize=(12, 6))
missing_df.sort_values("missing_rate", ascending=False).head(80).sort_values("missing_rate").plot(
    x="column", y="missing_rate", kind="barh", legend=False, ax=plt.gca(), color="#425e7a"
)
plt.xlabel("Missingness rate")
plt.title("Top 80 MASTER columns by missingness")
plt.tight_layout()
plt.savefig(FIG_DIR / "missingness_top80.png", dpi=170)
plt.show()
"""
        ),
        markdown_cell("## Phase 2 - Univariate Analysis By Feature Group"),
        code_cell(
            """
al_cols = cols(master, "AL_")
cat_cols = cols(master, "CAT_")
ek_cols = cols(master, "EK_")
aa_cols = ["AA_1", "AA_2"]

al_summary = master[al_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
al_summary["missing_rate"] = master[al_cols].isna().mean()
al_summary["skew"] = master[al_cols].skew(numeric_only=True)
al_summary["n_notna"] = master[al_cols].notna().sum()
al_summary["is_binary_01"] = [
    set(pd.to_numeric(master[c], errors="coerce").dropna().unique()).issubset({0, 1, 0.0, 1.0})
    and master[c].notna().any()
    for c in al_cols
]
al_summary.to_csv(TABLE_DIR / "phase2_al_distribution_summary.csv")

al_groups = al_summary.groupby(["n_notna", "is_binary_01"]).size().reset_index(name="columns")
al_groups = al_groups.sort_values("columns", ascending=False)
al_groups.to_csv(TABLE_DIR / "phase2_al_groups_by_n_notna.csv", index=False)
display(al_summary.head())
display(al_groups.head(20))
print("Binary AL columns:", int(al_summary["is_binary_01"].sum()))
"""
        ),
        code_cell(
            """
cat_rows = []
cat_label_rows = []
base_rate = master["Label"].mean()

for col in cat_cols:
    cat_text = master[col].astype("string")
    cat_rows.append({
        "column": col,
        "unique_values": int(master[col].nunique(dropna=True)),
        "missing_rate": float(master[col].isna().mean()),
        "composite_rate": float(cat_text.str.contains(r"&|,|\\|", na=False).mean()),
    })
    for value, count in master[col].fillna("missing").value_counts().items():
        subset = master[master[col].fillna("missing") == value]
        label_rate = float(subset["Label"].mean())
        cat_label_rows.append({
            "column": col,
            "value": value,
            "count": int(count),
            "label_rate": label_rate,
            "delta_from_base_rate": label_rate - base_rate,
            "deviates_gt_20pp": abs(label_rate - base_rate) > 0.20,
        })

cat_df = pd.DataFrame(cat_rows)
cat_label_df = pd.DataFrame(cat_label_rows)
cat_df.to_csv(TABLE_DIR / "phase2_cat_summary.csv", index=False)
cat_label_df.to_csv(TABLE_DIR / "phase2_cat_label_rates.csv", index=False)

display(cat_df)
cat_label_df.reindex(cat_label_df["delta_from_base_rate"].abs().sort_values(ascending=False).index).head(25)
"""
        ),
        code_cell(
            """
ek_summary = master[ek_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
ek_summary["missing_rate"] = master[ek_cols].isna().mean()
ek_summary["has_negative"] = (master[ek_cols] < 0).any()
ek_summary.to_csv(TABLE_DIR / "phase2_ek_summary.csv")
display(ek_summary)

ek_corr = master[ek_cols].corr(method="spearman")
ek_corr.to_csv(TABLE_DIR / "phase2_ek_spearman_correlation.csv")

plt.figure(figsize=(8, 7))
plt.imshow(ek_corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Spearman rho")
plt.xticks(range(len(ek_cols)), ek_cols, rotation=45, ha="right")
plt.yticks(range(len(ek_cols)), ek_cols)
plt.title("EK Spearman correlation")
plt.tight_layout()
plt.savefig(FIG_DIR / "ek_spearman_correlation.png", dpi=170)
plt.show()

ek_pairs = []
for i, c1 in enumerate(ek_cols):
    for c2 in ek_cols[i+1:]:
        rho = ek_corr.loc[c1, c2]
        if abs(rho) > 0.5:
            ek_pairs.append({"feature_1": c1, "feature_2": c2, "spearman_rho": rho})
pd.DataFrame(ek_pairs).sort_values("spearman_rho", ascending=False)
"""
        ),
        code_cell(
            """
aa_frequency = []
for col in aa_cols:
    for value, count in master[col].fillna("missing").value_counts().items():
        aa_frequency.append({"column": col, "value": value, "count": int(count)})
aa_frequency = pd.DataFrame(aa_frequency)
aa_frequency.to_csv(TABLE_DIR / "phase2_aa_frequency.csv", index=False)

aa_pair = master["AA_1"].astype(str) + ">" + master["AA_2"].astype(str)
aa_pairs = aa_pair.value_counts().reset_index()
aa_pairs.columns = ["aa_pair", "count"]
aa_pairs.to_csv(TABLE_DIR / "phase2_aa_pairs.csv", index=False)

standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
nonstandard = sorted((set(master["AA_1"].dropna().astype(str)) | set(master["AA_2"].dropna().astype(str))) - standard_aas)

display(aa_frequency.head(30))
display(aa_pairs.head(25))
print("Non-standard AA codes:", nonstandard)
"""
        ),
        markdown_cell("## Phase 3 - Bivariate Analysis: Features vs Label"),
        code_cell(
            """
aggregate = pd.DataFrame(index=master.index)
al_raw = [f"AL_{i}" for i in range(1, 27)]
aggregate["n_pops"] = master[al_raw].notna().sum(axis=1)
aggregate["n_nonmiss_AL"] = master[al_cols].notna().sum(axis=1)
aggregate["max_AF"] = master[al_raw].max(axis=1).fillna(0.0)
aggregate["min_AF_nz"] = master[al_raw].replace(0, np.nan).min(axis=1)
aggregate["EK7xEK9"] = master["EK_7"] * master["EK_9"]
aggregate["EK7_minus_EK9"] = master["EK_7"] - master["EK_9"]

numeric_for_auc = pd.concat([master[al_cols + ek_cols], aggregate], axis=1)

auc_rows = []
for feature in numeric_for_auc.columns:
    auc, pvalue, effect = auc_equivalent(numeric_for_auc[feature], master["Label"])
    auc_rows.append({
        "feature": feature,
        "auc_equivalent": auc,
        "auc_directional_strength": abs(auc - 0.5) + 0.5 if pd.notna(auc) else np.nan,
        "p_value": pvalue,
        "rank_biserial": effect,
        "n_notna": int(numeric_for_auc[feature].notna().sum()),
    })

auc_df = pd.DataFrame(auc_rows).sort_values("auc_directional_strength", ascending=False)
auc_df.to_csv(TABLE_DIR / "phase3_numeric_auc_ranked.csv", index=False)
display(auc_df.head(25))
display(auc_df.sort_values("auc_directional_strength").head(25))
"""
        ),
        code_cell(
            """
miss_signal_rows = []
for col in master.columns:
    rate = master[col].isna().mean()
    if 0 < rate < 1:
        miss = master[col].isna().astype(float)
        miss_signal_rows.append({
            "column": col,
            "missing_rate": rate,
            "label_rate_when_missing": float(master.loc[master[col].isna(), "Label"].mean()),
            "label_rate_when_present": float(master.loc[master[col].notna(), "Label"].mean()),
            "missing_label_corr": float(miss.corr(master["Label"])),
        })

miss_signal_df = pd.DataFrame(miss_signal_rows).sort_values(
    "missing_label_corr", key=lambda s: s.abs(), ascending=False
)
miss_signal_df.to_csv(TABLE_DIR / "phase3_missingness_as_feature.csv", index=False)
miss_signal_df.head(25)
"""
        ),
        code_cell(
            """
top10 = auc_df["feature"].head(10).tolist()
plot_df = master.join(aggregate)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for ax, feature in zip(axes.ravel(), top10):
    data_to_plot = [
        pd.to_numeric(plot_df.loc[plot_df["Label"] == label, feature], errors="coerce").dropna()
        for label in [0, 1]
    ]
    ax.boxplot(data_to_plot, labels=["Benign", "Pathogenic"], showfliers=False)
    ax.set_title(feature)
plt.tight_layout()
plt.savefig(FIG_DIR / "top10_numeric_boxplots.png", dpi=170)
plt.show()
"""
        ),
        markdown_cell("## Phase 4 - Multivariate And Structural Analysis"),
        code_cell(
            """
triplet_rows = []
for start in range(27, 334, 3):
    triple = [f"AL_{i}" for i in range(start, min(start + 3, 335)) if f"AL_{i}" in master.columns]
    if len(triple) == 3:
        corr = master[triple].corr(method="spearman")
        vals = [corr.iloc[0, 1], corr.iloc[0, 2], corr.iloc[1, 2]]
        mean_abs_rho = np.nan if pd.isna(vals).all() else float(np.nanmean(np.abs(vals)))
        triplet_rows.append({
            "columns": ",".join(triple),
            "rho_12": vals[0],
            "rho_13": vals[1],
            "rho_23": vals[2],
            "mean_abs_rho": mean_abs_rho,
        })

triplet_df = pd.DataFrame(triplet_rows)
triplet_df.to_csv(TABLE_DIR / "phase4_al_triplet_correlations.csv", index=False)
display(triplet_df.sort_values("mean_abs_rho").head(15))
display(triplet_df.sort_values("mean_abs_rho", ascending=False).head(15))
"""
        ),
        code_cell(
            """
cross = master[ek_cols].join(aggregate).corr(method="spearman")
cross.to_csv(TABLE_DIR / "phase4_cross_group_spearman.csv")

high_cross = []
for c1 in ek_cols:
    for c2 in aggregate.columns:
        rho = cross.loc[c1, c2]
        if pd.notna(rho) and abs(rho) > 0.5:
            high_cross.append({"feature_1": c1, "feature_2": c2, "spearman_rho": rho})

high_cross_df = pd.DataFrame(high_cross)
high_cross_df.to_csv(TABLE_DIR / "phase4_high_cross_group_correlations.csv", index=False)
high_cross_df
"""
        ),
        code_cell(
            """
x_al = master[al_cols].apply(pd.to_numeric, errors="coerce")
x_al = SimpleImputer(strategy="median").fit_transform(x_al)
x_al = StandardScaler().fit_transform(x_al)

pca = PCA(n_components=20, random_state=42)
pca.fit(x_al)
pca_df = pd.DataFrame({
    "component": [f"PC{i}" for i in range(1, 21)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
})
pca_df.to_csv(TABLE_DIR / "phase4_al_pca.csv", index=False)
display(pca_df)

plt.figure(figsize=(9, 5))
plt.plot(range(1, 21), pca_df["cumulative_explained_variance"], marker="o")
plt.xlabel("Number of AL PCs")
plt.ylabel("Cumulative explained variance")
plt.title("AL block PCA")
plt.tight_layout()
plt.savefig(FIG_DIR / "al_pca_cumulative_variance.png", dpi=170)
plt.show()
"""
        ),
        code_cell(
            """
leakage_df = auc_df[auc_df["auc_directional_strength"] > 0.85]
leakage_df.to_csv(TABLE_DIR / "phase4_leakage_suspects.csv", index=False)
if leakage_df.empty:
    print("No individual feature exceeds directional AUC strength 0.85.")
else:
    display(leakage_df)
"""
        ),
        markdown_cell("## Phase 5 - Cross-Dataset Comparison"),
        code_cell(
            """
top10_features = auc_df["feature"].head(10).tolist()
shift_rows = []

for name, df in data.items():
    agg = pd.DataFrame(index=df.index)
    agg["n_pops"] = df[al_raw].notna().sum(axis=1)
    agg["n_nonmiss_AL"] = df[al_cols].notna().sum(axis=1)
    agg["max_AF"] = df[al_raw].max(axis=1).fillna(0.0)
    agg["min_AF_nz"] = df[al_raw].replace(0, np.nan).min(axis=1)
    agg["EK7xEK9"] = df["EK_7"] * df["EK_9"]
    agg["EK7_minus_EK9"] = df["EK_7"] - df["EK_9"]
    merged = df.join(agg)
    for feature in top10_features:
        if feature in merged.columns:
            shift_rows.append({
                "dataset": name,
                "feature": feature,
                "mean": float(pd.to_numeric(merged[feature], errors="coerce").mean()),
                "median": float(pd.to_numeric(merged[feature], errors="coerce").median()),
                "pathogenic_mean": float(pd.to_numeric(merged.loc[merged["Label"] == 1, feature], errors="coerce").mean()),
                "benign_mean": float(pd.to_numeric(merged.loc[merged["Label"] == 0, feature], errors="coerce").mean()),
                "missing_rate": float(merged[feature].isna().mean()),
            })

shift_df = pd.DataFrame(shift_rows)
shift_df.to_csv(TABLE_DIR / "phase5_top10_cross_dataset_shift.csv", index=False)
shift_df.head(40)
"""
        ),
        code_cell(
            """
panel_unique_rows = []
numeric_cols = al_cols + ek_cols

for name in panel_names:
    df = data[name]
    unique = df[~df["Variant_ID"].isin(master_ids)]
    shared = df[df["Variant_ID"].isin(master_ids)]
    for subset_name, subset in [("unique", unique), ("shared_with_master", shared)]:
        panel_unique_rows.append({
            "panel": name,
            "subset": subset_name,
            "n": len(subset),
            "pathogenic_rate": float(subset["Label"].mean()),
            "overall_missing_rate": float(subset.isna().mean().mean()),
            "numeric_range_mean": float(
                subset[numeric_cols].apply(
                    lambda s: pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min()
                ).mean()
            ),
        })

panel_unique_df = pd.DataFrame(panel_unique_rows)
panel_unique_df.to_csv(TABLE_DIR / "phase5_panel_unique_vs_shared.csv", index=False)
panel_unique_df
"""
        ),
        markdown_cell(
            """
### Panel Summary

KANSER unique subset has a much lower pathogenic rate than KANSER variants shared
with MASTER, indicating a major distribution shift. PAH and CFTR unique subsets
are smaller, but also differ strongly from shared subsets. This supports treating
panel-unique variants as a harder generalization check.
"""
        ),
        markdown_cell("## Phase 6 - ACMG Evidence Interpretation"),
        code_cell(
            """
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
    acmg_rows.append({
        "feature": f,
        "auc_equivalent": row["auc_equivalent"],
        "proposed_acmg_category": criterion,
        "justification": meaning,
        "confidence": confidence,
    })

acmg_df = pd.DataFrame(acmg_rows)
acmg_df.to_csv(TABLE_DIR / "phase6_acmg_interpretation.csv", index=False)
acmg_df
"""
        ),
        markdown_cell(
            """
## Output Files

All notebook-generated tables are written to `reports/eda_notebook_outputs/tables`.
All notebook-generated figures are written to `reports/eda_notebook_outputs/figures`.
"""
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Notebook written to: {NOTEBOOK_PATH.resolve()}")


if __name__ == "__main__":
    main()
