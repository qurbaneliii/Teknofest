from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "reports" / "eda"
EDA_TABLES = EDA_DIR / "tables"
EDA_FIGURES = EDA_DIR / "figures"
MODEL_TABLES = ROOT / "reports" / "tables"
OUT_REPORT = EDA_DIR / "EDA_MODEL_BRIDGE.md"


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def build_feature_alignment() -> pd.DataFrame:
    numeric = pd.read_csv(EDA_TABLES / "phase3_numeric_auc_ranked.csv")
    shap = pd.read_csv(MODEL_TABLES / "feature_importance.csv")
    shap["shap_rank"] = shap["mean_abs_shap"].rank(method="min", ascending=False).astype(int)
    numeric["eda_rank"] = numeric["auc_directional_strength"].rank(method="min", ascending=False).astype(int)
    aligned = numeric.merge(shap, on="feature", how="left")
    aligned["in_final_model_shap"] = aligned["mean_abs_shap"].notna()
    return aligned.sort_values(["in_final_model_shap", "auc_directional_strength"], ascending=[False, False])


def build_panel_alignment() -> pd.DataFrame:
    shift = pd.read_csv(EDA_TABLES / "phase5_panel_unique_vs_shared.csv")
    panel = pd.read_csv(MODEL_TABLES / "panel_generalization_results.csv")
    unique = shift[shift["subset"].eq("unique")].copy()
    unique["dataset"] = unique["panel"] + "_unique"
    return unique.merge(panel, on=["dataset", "n"], how="left").sort_values("panel")


def build_decision_bridge() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eda_finding": "Panel-shared variants have much higher pathogenic rates than panel-unique variants.",
                "implemented_response": "MASTER variants shared with panels are excluded from validation folds; panel-unique subsets are evaluated separately.",
                "verification_artifact": "reports/tables/validation_split_diagnostics.csv; reports/tables/panel_generalization_results.csv",
            },
            {
                "eda_finding": "Missingness is label-associated for AL blocks and has structured availability groups.",
                "implemented_response": "Feature engineering includes missingness flags, missingness PCA, non-missing counts, and leakage-safe fold fitting.",
                "verification_artifact": "artifacts/metrics/feature_list.json; reports/eda/tables/phase3_missingness_as_feature.csv",
            },
            {
                "eda_finding": "EK_7, EK_9, EK_3, and EK7xEK9 are strong univariate pathogenicity signals.",
                "implemented_response": "Main feature set preserves EK raw values and adds EK interactions/evidence aggregates.",
                "verification_artifact": "reports/tables/feature_importance.csv; reports/tables/main_model_cv_results.csv",
            },
            {
                "eda_finding": "Population-frequency style AL aggregates and n_pops separate benign/pathogenic variants.",
                "implemented_response": "ACMG-inspired BA1/BS1/PM2/BS2 proxy features are generated and mapped to clinical evidence categories.",
                "verification_artifact": "reports/tables/acmg_feature_mapping.csv; reports/eda/tables/phase6_acmg_interpretation.csv",
            },
            {
                "eda_finding": "No single numeric leakage suspect exceeded AUC-equivalent directional strength 0.85.",
                "implemented_response": "No feature was removed solely as a leakage proxy, but validation remains contamination-aware and panel-unique.",
                "verification_artifact": "reports/eda/tables/phase4_leakage_suspects.csv",
            },
            {
                "eda_finding": "Panel-unique subsets have distribution shift and limited sample sizes.",
                "implemented_response": "Panel metrics are reported with bootstrap confidence intervals instead of only point estimates.",
                "verification_artifact": "reports/tables/panel_unique_bootstrap_ci.csv",
            },
        ]
    )


def plot_alignment(aligned: pd.DataFrame) -> None:
    plot_df = aligned[aligned["in_final_model_shap"]].head(15).sort_values("mean_abs_shap")
    if plot_df.empty:
        return
    EDA_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#4f6f7f")
    plt.xlabel("Mean absolute SHAP value")
    plt.title("EDA top numeric signals present in final model")
    plt.tight_layout()
    plt.savefig(EDA_FIGURES / "eda_model_feature_alignment.png", dpi=170)
    plt.close()


def write_report(decisions: pd.DataFrame, aligned: pd.DataFrame, panel: pd.DataFrame) -> None:
    threshold = pd.read_csv(MODEL_TABLES / "threshold_results.csv")
    f1_row = threshold[threshold["threshold_name"].eq("f1_macro_opt")].iloc[0]
    trials = pd.read_csv(ROOT / "reports" / "master_prompt" / "lgbm_optuna_trials_resumable.csv")
    complete = trials[trials["state"].eq("COMPLETE")]

    aligned_out = aligned[
        [
            "feature",
            "eda_rank",
            "auc_equivalent",
            "auc_directional_strength",
            "p_value",
            "mean_abs_shap",
            "shap_rank",
            "in_final_model_shap",
        ]
    ].copy()
    aligned_out.to_csv(EDA_TABLES / "eda_model_feature_alignment.csv", index=False)
    panel.to_csv(EDA_TABLES / "eda_panel_model_alignment.csv", index=False)
    decisions.to_csv(EDA_TABLES / "eda_to_model_decision_bridge.csv", index=False)

    report = f"""# EDA To Model Bridge

This report connects the exploratory data analysis findings to concrete modeling decisions and verification artifacts.

## Decision Bridge

{markdown_table(decisions, max_rows=20)}

## Feature Alignment

The table below joins the strongest EDA numeric signals with final model SHAP importance. Missing SHAP values mean the raw EDA feature was not present under the same name in the final numeric model, often because it was transformed into an engineered aggregate.

{markdown_table(aligned_out.head(25), max_rows=25)}

Figure: `reports/eda/figures/eda_model_feature_alignment.png`

## Panel Shift To Model Results

{markdown_table(panel, max_rows=10)}

## Final Tuned Model Context

- Optuna complete trials: {len(complete)}
- Best Optuna contamination-aware CV AUC: {complete["mean_auc"].max():.6f}
- F1-macro optimized threshold: {float(f1_row["threshold"]):.6f}
- F1-macro at optimized threshold: {float(f1_row["f1_macro"]):.6f}

## Conclusion

The EDA findings are directly represented in the final pipeline: contamination-aware validation handles overlap risk, missingness-derived features are preserved because missingness is predictive, EK and population-frequency signals are retained, and panel-unique performance is reported separately due to distribution shift.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    EDA_TABLES.mkdir(parents=True, exist_ok=True)
    decisions = build_decision_bridge()
    aligned = build_feature_alignment()
    panel = build_panel_alignment()
    plot_alignment(aligned)
    write_report(decisions, aligned, panel)
    print(f"Wrote {OUT_REPORT.resolve()}")


if __name__ == "__main__":
    main()
