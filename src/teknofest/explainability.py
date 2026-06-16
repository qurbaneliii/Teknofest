from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix

from teknofest.data_prep import PreparedData
from teknofest.training import model_columns


ACMG_FEATURE_MAP = pd.DataFrame(
    [
        {
            "feature": "max_AF",
            "shap_direction": "Negative when high",
            "acmg_criterion": "BA1",
            "clinical_meaning": "Allele frequency above benign threshold.",
        },
        {
            "feature": "n_pops",
            "shap_direction": "Positive when zero",
            "acmg_criterion": "PM2",
            "clinical_meaning": "Absent from population databases.",
        },
        {
            "feature": "EK_7",
            "shap_direction": "Positive when high",
            "acmg_criterion": "PP3/BP4",
            "clinical_meaning": "Highly conserved position supports pathogenicity.",
        },
        {
            "feature": "EK_9",
            "shap_direction": "Positive when high",
            "acmg_criterion": "PP3/BP4",
            "clinical_meaning": "Evolutionary model predicts pathogenic effect.",
        },
        {
            "feature": "cat1_multipop",
            "shap_direction": "Positive when true",
            "acmg_criterion": "PM2 strong",
            "clinical_meaning": "Rare across multiple population sources.",
        },
        {
            "feature": "cat6_segdup",
            "shap_direction": "Negative when true",
            "acmg_criterion": "BP7 proxy",
            "clinical_meaning": "Low-confidence segmental duplication region.",
        },
        {
            "feature": "aa_involves_G",
            "shap_direction": "Positive when true",
            "acmg_criterion": "PS1/PM5",
            "clinical_meaning": "Glycine substitutions can disrupt structure.",
        },
    ]
)


SHAP_GROUPS = {
    "Conservation/Evolution": ["EK_7", "EK_9", "EK_3", "EK7xEK9", "EK7_minus_EK9"],
    "Population Frequency": [
        "max_AF",
        "min_AF_nz",
        "log_max_AF",
        "log_min_AF",
        "n_pops",
        "n_nonmiss_AL",
        "BA1_flag",
        "BS1_flag",
        "PM2_flag",
        "BS2_proxy",
        "cat1_multipop",
    ],
    "Computational Evidence": [
        "EK_1",
        "EK_2",
        "EK_4",
        "EK_5",
        "EK_6",
        "EK_8",
        "n_EK_path",
        "n_EK_ben",
        "EK_net_evidence",
    ],
    "Amino Acid Chemistry": [
        "aa_involves_G",
        "aa_involves_P",
        "aa_involves_C",
        "aa_involves_R",
        "aa_class_changed",
        "blosum62_approx",
        "aa_change_te",
    ],
    "Missingness": ["miss_AL1_6", "miss_AL7_15", "miss_AL16_25", "miss_AL27_38", "miss_EK3"],
}


def load_model_artifacts(model_dir: Path):
    engineer = joblib.load(model_dir / "feature_engineer.joblib")
    model = joblib.load(model_dir / "lightgbm_final.joblib")
    columns = (model_dir / "model_columns.txt").read_text(encoding="utf-8").splitlines()
    return engineer, model, columns


def shap_values_for_master(prepared: PreparedData, model_dir: Path, sample_size: int | None = None):
    engineer, model, columns = load_model_artifacts(model_dir)
    master = engineer.transform(prepared.master)
    x = master.reindex(columns=columns)
    y = master["Label"].copy()
    ids = master["Variant_ID"].copy()

    if sample_size is not None and sample_size < len(x):
        sampled = x.sample(n=sample_size, random_state=42).index
        x = x.loc[sampled]
        y = y.loc[sampled]
        ids = ids.loc[sampled]

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]
    return x, y, ids, model, values, explainer.expected_value


def save_global_importance(values: np.ndarray, x: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": x.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(out_dir / "shap_global_importance.csv", index=False)

    top = importance.head(25).sort_values("mean_abs_shap")
    plt.figure(figsize=(9, 8))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#315f72")
    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=180)
    plt.close()
    return importance


def save_group_importance(values: np.ndarray, x: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for group, features in SHAP_GROUPS.items():
        present = [feature for feature in features if feature in x.columns]
        if not present:
            continue
        idx = [x.columns.get_loc(feature) for feature in present]
        rows.append(
            {
                "group": group,
                "features_present": len(present),
                "mean_abs_shap": float(np.abs(values[:, idx]).sum(axis=1).mean()),
            }
        )
    group_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    group_df.to_csv(out_dir / "shap_group_importance.csv", index=False)
    return group_df


def save_plots(values: np.ndarray, x: pd.DataFrame, out_dir: Path) -> None:
    shap.summary_plot(values, x, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    if "EK_7" in x.columns and "EK_9" in x.columns:
        shap.dependence_plot("EK_7", values, x, interaction_index="EK_9", show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_dependence_EK7_by_EK9.png", dpi=180, bbox_inches="tight")
        plt.close()

    if "n_pops" in x.columns and "max_AF" in x.columns:
        shap.dependence_plot("n_pops", values, x, interaction_index="max_AF", show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_dependence_npops_by_maxAF.png", dpi=180, bbox_inches="tight")
        plt.close()


def save_case_waterfalls(
    values: np.ndarray,
    x: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    model,
    expected_value,
    out_dir: Path,
    threshold: float = 0.5,
    per_group: int = 5,
    prefix: str = "",
) -> pd.DataFrame:
    probs = model.predict_proba(x)[:, 1]
    pred = (probs >= threshold).astype(int)
    case_df = pd.DataFrame(
        {
            "Variant_ID": ids.to_numpy(),
            "Label": y.to_numpy(),
            "prediction": pred,
            "probability": probs,
        },
        index=x.index,
    )
    case_df["case_type"] = np.select(
        [
            (case_df["Label"] == 1) & (case_df["prediction"] == 1),
            (case_df["Label"] == 0) & (case_df["prediction"] == 0),
            (case_df["Label"] == 0) & (case_df["prediction"] == 1),
            (case_df["Label"] == 1) & (case_df["prediction"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="unknown",
    )
    selected = []
    for case_type in ["TP", "TN", "FP", "FN"]:
        subset = case_df[case_df["case_type"] == case_type].copy()
        subset["confidence"] = (subset["probability"] - 0.5).abs()
        selected.extend(subset.sort_values("confidence", ascending=False).head(per_group).index)

    base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    for rank, idx in enumerate(selected, start=1):
        row_pos = x.index.get_loc(idx)
        explanation = shap.Explanation(
            values=values[row_pos],
            base_values=base_value,
            data=x.iloc[row_pos].to_numpy(),
            feature_names=list(x.columns),
        )
        shap.plots.waterfall(explanation, max_display=15, show=False)
        case_type = case_df.loc[idx, "case_type"]
        plt.tight_layout()
        plt.savefig(
            out_dir / f"shap_waterfall_{prefix}{rank:02d}_{case_type}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    case_df.loc[selected].to_csv(out_dir / f"shap_waterfall_{prefix}cases.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y, pred),
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(
        out_dir / f"shap_case_{prefix}confusion_matrix.csv"
    )
    return case_df.loc[selected]


def run_explainability(
    prepared: PreparedData,
    model_dir: Path,
    out_dir: Path,
    sample_size: int | None = 1000,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, ids, model, values, expected_value = shap_values_for_master(
        prepared,
        model_dir,
        sample_size=sample_size,
    )
    save_global_importance(values, x, out_dir)
    save_group_importance(values, x, out_dir)
    save_plots(values, x, out_dir)
    master_cases = save_case_waterfalls(values, x, y, ids, model, expected_value, out_dir)

    missing_case_types = {"TP", "TN", "FP", "FN"} - set(master_cases["case_type"])
    if missing_case_types:
        engineer, model, columns = load_model_artifacts(model_dir)
        panel_raw = pd.concat(
            [
                prepared.kanser_unique.assign(_source="KANSER_unique"),
                prepared.pah_unique.assign(_source="PAH_unique"),
                prepared.cftr_unique.assign(_source="CFTR_unique"),
            ],
            axis=0,
            ignore_index=True,
        )
        panel = engineer.transform(panel_raw)
        panel_x = panel.reindex(columns=columns)
        panel_y = panel["Label"]
        panel_ids = panel["_source"].astype(str) + ":" + panel["Variant_ID"].astype(str)
        panel_explainer = shap.TreeExplainer(model)
        panel_values = panel_explainer.shap_values(panel_x)
        if isinstance(panel_values, list):
            panel_values = panel_values[1]
        if panel_values.ndim == 3:
            panel_values = panel_values[:, :, 1]
        save_case_waterfalls(
            panel_values,
            panel_x,
            panel_y,
            panel_ids,
            model,
            panel_explainer.expected_value,
            out_dir,
            prefix="panel_unique_",
        )
    ACMG_FEATURE_MAP.to_csv(out_dir / "acmg_feature_mapping.csv", index=False)
