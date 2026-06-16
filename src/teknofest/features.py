from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


HYDROPHOBIC = set("VILMFYWC")
POS_CHARGED = set("RKH")
NEG_CHARGED = set("DE")
SPECIAL = set("GP")

BLOSUM62_KEY = {
    ("G", "R"): -2,
    ("G", "P"): 0,
    ("C", "R"): -3,
    ("C", "Y"): -2,
    ("L", "P"): -3,
    ("R", "W"): -3,
    ("R", "Q"): 1,
    ("E", "K"): 1,
    ("A", "T"): 0,
    ("V", "M"): 1,
    ("S", "T"): 1,
    ("N", "D"): 1,
}


def detect_binary_al_cols(df: pd.DataFrame, al_cols: list[str]) -> list[str]:
    flag_cols: list[str] = []
    for col in al_cols:
        values = set(pd.to_numeric(df[col], errors="coerce").dropna().unique())
        if values and values.issubset({0, 1, 0.0, 1.0}):
            flag_cols.append(col)
    return flag_cols


def aa_class(aa: object) -> str:
    if pd.isna(aa):
        return "unknown"
    aa_str = str(aa)
    if aa_str in HYDROPHOBIC:
        return "hydrophobic"
    if aa_str in POS_CHARGED:
        return "pos_charged"
    if aa_str in NEG_CHARGED:
        return "neg_charged"
    if aa_str in SPECIAL:
        return "special"
    return "polar"


def blosum62_approx(row: pd.Series) -> int:
    key = (str(row["AA_1"]), str(row["AA_2"]))
    return BLOSUM62_KEY.get(key, BLOSUM62_KEY.get(key[::-1], 0))


@dataclass
class FeatureEngineer:
    al_cols: list[str]
    al_raw: list[str]
    flag_cols: list[str]
    n_missing_pcs: int = 10
    cat1_alpha: float = 30.0
    aa_alpha: float = 10.0
    pca_: PCA | None = field(default=None, init=False)
    cat1_encoder_: dict[str, float] = field(default_factory=dict, init=False)
    aa_encoder_: dict[str, float] = field(default_factory=dict, init=False)
    cat2_columns_: list[str] = field(default_factory=list, init=False)
    global_mean_: float = field(default=0.73, init=False)

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        if "Label" in df.columns:
            self.global_mean_ = float(df["Label"].mean())

        n_components = min(self.n_missing_pcs, len(self.al_cols), len(df))
        self.pca_ = PCA(n_components=n_components, random_state=42)
        self.pca_.fit(df[self.al_cols].isna().astype(float))

        if "Label" in df.columns:
            counts = df.groupby("CAT_1", dropna=False)["Label"].agg(["sum", "count"])
            self.cat1_encoder_ = (
                (counts["sum"] + self.cat1_alpha * self.global_mean_)
                / (counts["count"] + self.cat1_alpha)
            ).to_dict()
            self.cat1_encoder_["__global__"] = self.global_mean_

            aa_change = df["AA_1"].astype(str) + df["AA_2"].astype(str)
            aa_counts = pd.DataFrame(
                {"_aa_change": aa_change, "Label": df["Label"]}
            ).groupby("_aa_change")["Label"].agg(["sum", "count"])
            self.aa_encoder_ = (
                (aa_counts["sum"] + self.aa_alpha * self.global_mean_)
                / (aa_counts["count"] + self.aa_alpha)
            ).to_dict()
            self.aa_encoder_["__global__"] = self.global_mean_

        self.cat2_columns_ = sorted(
            pd.get_dummies(df["CAT_2"].fillna("missing"), prefix="cat2").columns
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pca_ is None:
            raise RuntimeError("FeatureEngineer must be fit before transform.")

        d = df.copy()

        d["miss_AL1_6"] = d["AL_1"].isna().astype(float)
        d["miss_AL7_15"] = d["AL_7"].isna().astype(float)
        d["miss_AL16_25"] = d["AL_16"].isna().astype(float)
        d["miss_AL27_38"] = d["AL_27"].isna().astype(float)
        d["miss_EK3"] = d["EK_3"].isna().astype(float)

        miss_pcs = self.pca_.transform(d[self.al_cols].isna().astype(float))
        for idx in range(miss_pcs.shape[1]):
            d[f"miss_pc{idx + 1}"] = miss_pcs[:, idx]

        d["n_pops"] = d[self.al_raw].notna().sum(axis=1).astype(float)
        d["n_nonmiss_AL"] = d[self.al_cols].notna().sum(axis=1).astype(float)
        d["max_AF"] = d[self.al_raw].max(axis=1).fillna(0.0)
        d["min_AF_nz"] = d[self.al_raw].replace(0, np.nan).min(axis=1)
        d["log_max_AF"] = np.log10(d["max_AF"] + 1e-15)
        d["log_min_AF"] = np.log10(d["min_AF_nz"].fillna(1e-15) + 1e-15)
        d["BA1_flag"] = (d["max_AF"] > 0.05).astype(float)
        d["BS1_flag"] = (d["max_AF"] > 0.01).astype(float)
        d["PM2_flag"] = (d["n_pops"] == 0).astype(float)
        d["BS2_proxy"] = (d["n_pops"] >= 20).astype(float)
        d["n_pops_squared"] = d["n_pops"] ** 2
        d["npops_tier_0"] = (d["n_pops"] == 0).astype(float)
        d["npops_tier_10"] = (d["n_pops"] == 10).astype(float)
        d["npops_tier_26"] = (d["n_pops"] == 26).astype(float)

        d["sum_flags"] = d[self.flag_cols].sum(axis=1) if self.flag_cols else 0.0
        d["n_valid_flags"] = (
            d[self.flag_cols].notna().sum(axis=1).clip(lower=1)
            if self.flag_cols
            else 1.0
        )
        d["frac_flags"] = d["sum_flags"] / d["n_valid_flags"]

        d["EK7xEK9"] = d["EK_7"] * d["EK_9"]
        d["EK2xEK4"] = d["EK_2"] * d["EK_4"]
        d["EK4xEK6"] = d["EK_4"] * d["EK_6"]
        d["EK7_minus_EK9"] = d["EK_7"] - d["EK_9"]
        d["max_prob_EK"] = d[["EK_4", "EK_5", "EK_6"]].max(axis=1)
        d["n_EK_path"] = (
            (d["EK_2"] > 0).astype(float)
            + (d["EK_7"] > 5.0).astype(float)
            + (d["EK_9"] > 7.0).astype(float)
            + (d["EK_4"] > 0.7).astype(float)
            + (d["EK_6"] > 0.7).astype(float)
            + (d["EK_3"] > 2.0).astype(float)
        )
        d["n_EK_ben"] = (
            (d["EK_2"] < 0).astype(float)
            + (d["EK_7"] < 1.0).astype(float)
            + (d["EK_9"] < 3.0).astype(float)
            + (d["EK_4"] < 0.3).astype(float)
            + (d["EK_6"] < 0.3).astype(float)
        )
        d["EK_net_evidence"] = d["n_EK_path"] - d["n_EK_ben"]

        cat1 = d["CAT_1"].astype("string")
        d["cat1_multipop"] = cat1.str.contains("&", na=False).astype(float)
        d["cat1_AFR"] = cat1.str.contains("AFR", na=False).astype(float)
        d["cat1_NFE"] = cat1.str.contains("NFE", na=False).astype(float)
        d["cat1_gnomADe"] = cat1.str.contains("gnomADe", na=False).astype(float)
        d["cat1_gnomADg"] = cat1.str.contains("gnomADg", na=False).astype(float)
        d["cat1_te"] = d["CAT_1"].map(self.cat1_encoder_).fillna(
            self.cat1_encoder_.get("__global__", self.global_mean_)
        )

        cat2_dummies = pd.get_dummies(d["CAT_2"].fillna("missing"), prefix="cat2")
        cat2_dummies = cat2_dummies.reindex(columns=self.cat2_columns_, fill_value=False)
        d = pd.concat([d, cat2_dummies.astype(float)], axis=1)

        d["cat6_has_filter"] = d["CAT_6"].notna().astype(float)
        d["cat6_segdup"] = (d["CAT_6"] == "segdup").astype(float)
        d["cat6_lcr"] = (d["CAT_6"] == "lcr").astype(float)
        d["geno_consistent"] = (
            (d["CAT_3"] == d["CAT_4"]) & (d["CAT_4"] == d["CAT_5"])
        ).astype(float)
        d["geno_missing"] = (d["CAT_3"] == "./.").astype(float)

        d["aa1_class"] = d["AA_1"].apply(aa_class)
        d["aa2_class"] = d["AA_2"].apply(aa_class)
        d["aa_class_change"] = d["aa1_class"] + "_to_" + d["aa2_class"]
        d["aa_involves_G"] = ((d["AA_1"] == "G") | (d["AA_2"] == "G")).astype(float)
        d["aa_involves_P"] = ((d["AA_1"] == "P") | (d["AA_2"] == "P")).astype(float)
        d["aa_involves_C"] = ((d["AA_1"] == "C") | (d["AA_2"] == "C")).astype(float)
        d["aa_involves_R"] = ((d["AA_1"] == "R") | (d["AA_2"] == "R")).astype(float)
        d["aa_to_special"] = (d["aa2_class"] == "special").astype(float)
        d["aa_from_special"] = (d["aa1_class"] == "special").astype(float)
        d["aa_class_changed"] = (d["aa1_class"] != d["aa2_class"]).astype(float)
        d["blosum62_approx"] = d.apply(blosum62_approx, axis=1)

        aa_change = d["AA_1"].astype(str) + d["AA_2"].astype(str)
        d["aa_change_te"] = aa_change.map(self.aa_encoder_).fillna(
            self.aa_encoder_.get("__global__", self.global_mean_)
        )
        return d

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
