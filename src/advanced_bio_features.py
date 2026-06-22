from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


AMINO_ACIDS = tuple("ARNDCQEGHILKMFPSTWYV")
AA_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "polarity": 8.1, "weight": 89.09, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "aliphatic"},
    "R": {"hydrophobicity": -4.5, "polarity": 10.5, "weight": 174.20, "charge": 1, "aromatic": 0, "sulfur": 0, "class": "basic"},
    "N": {"hydrophobicity": -3.5, "polarity": 11.6, "weight": 132.12, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "polar"},
    "D": {"hydrophobicity": -3.5, "polarity": 13.0, "weight": 133.10, "charge": -1, "aromatic": 0, "sulfur": 0, "class": "acidic"},
    "C": {"hydrophobicity": 2.5, "polarity": 5.5, "weight": 121.16, "charge": 0, "aromatic": 0, "sulfur": 1, "class": "sulfur"},
    "Q": {"hydrophobicity": -3.5, "polarity": 10.5, "weight": 146.15, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "polar"},
    "E": {"hydrophobicity": -3.5, "polarity": 12.3, "weight": 147.13, "charge": -1, "aromatic": 0, "sulfur": 0, "class": "acidic"},
    "G": {"hydrophobicity": -0.4, "polarity": 9.0, "weight": 75.07, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "special"},
    "H": {"hydrophobicity": -3.2, "polarity": 10.4, "weight": 155.16, "charge": 1, "aromatic": 1, "sulfur": 0, "class": "basic"},
    "I": {"hydrophobicity": 4.5, "polarity": 5.2, "weight": 131.18, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "aliphatic"},
    "L": {"hydrophobicity": 3.8, "polarity": 4.9, "weight": 131.18, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "aliphatic"},
    "K": {"hydrophobicity": -3.9, "polarity": 11.3, "weight": 146.19, "charge": 1, "aromatic": 0, "sulfur": 0, "class": "basic"},
    "M": {"hydrophobicity": 1.9, "polarity": 5.7, "weight": 149.21, "charge": 0, "aromatic": 0, "sulfur": 1, "class": "sulfur"},
    "F": {"hydrophobicity": 2.8, "polarity": 5.2, "weight": 165.19, "charge": 0, "aromatic": 1, "sulfur": 0, "class": "aromatic"},
    "P": {"hydrophobicity": -1.6, "polarity": 8.0, "weight": 115.13, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "special"},
    "S": {"hydrophobicity": -0.8, "polarity": 9.2, "weight": 105.09, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "polar"},
    "T": {"hydrophobicity": -0.7, "polarity": 8.6, "weight": 119.12, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "polar"},
    "W": {"hydrophobicity": -0.9, "polarity": 5.4, "weight": 204.23, "charge": 0, "aromatic": 1, "sulfur": 0, "class": "aromatic"},
    "Y": {"hydrophobicity": -1.3, "polarity": 6.2, "weight": 181.19, "charge": 0, "aromatic": 1, "sulfur": 0, "class": "aromatic"},
    "V": {"hydrophobicity": 4.2, "polarity": 5.9, "weight": 117.15, "charge": 0, "aromatic": 0, "sulfur": 0, "class": "aliphatic"},
}
AA_CLASS_ORDER = ("acidic", "aliphatic", "aromatic", "basic", "polar", "special", "sulfur", "unknown")
AA_CLASS_TRANSITION_CODES = {
    f"{source}_to_{target}": float(index)
    for index, (source, target) in enumerate(
        (source, target) for source in AA_CLASS_ORDER for target in AA_CLASS_ORDER
    )
}

# Standard BLOSUM62 matrix, stored compactly to keep substitution handling deterministic.
_BLOSUM62_ROWS = {
    "A": "4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0",
    "R": "-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3",
    "N": "-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3",
    "D": "-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3",
    "C": "0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1",
    "Q": "-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2",
    "E": "-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2",
    "G": "0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3",
    "H": "-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3",
    "I": "-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3",
    "L": "-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1",
    "K": "-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2",
    "M": "-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1",
    "F": "-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1",
    "P": "-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2",
    "S": "1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2",
    "T": "0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0",
    "W": "-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3",
    "Y": "-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1",
    "V": "0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4",
}
BLOSUM62 = {aa: dict(zip(AMINO_ACIDS, map(int, row.split()))) for aa, row in _BLOSUM62_ROWS.items()}

# Grantham's original 20 amino-acid distance matrix in the same AA order.
_GRANTHAM_ROWS = {
    "A": "0 112 111 126 195 91 107 60 86 94 96 106 84 113 27 99 58 148 112 64",
    "R": "112 0 86 96 180 43 54 125 29 97 102 26 91 97 103 110 71 101 77 96",
    "N": "111 86 0 23 139 46 42 80 68 149 153 94 142 158 91 46 65 174 143 133",
    "D": "126 96 23 0 154 61 45 94 81 168 172 101 160 177 108 65 85 181 160 152",
    "C": "195 180 139 154 0 154 170 159 174 198 198 202 196 205 169 112 149 215 194 192",
    "Q": "91 43 46 61 154 0 29 87 24 109 113 53 101 116 76 68 42 130 99 96",
    "E": "107 54 42 45 170 29 0 98 40 134 138 56 126 140 93 80 65 152 122 121",
    "G": "60 125 80 94 159 87 98 0 98 135 138 127 127 153 42 56 59 184 147 109",
    "H": "86 29 68 81 174 24 40 98 0 94 99 32 87 100 77 89 47 115 83 84",
    "I": "94 97 149 168 198 109 134 135 94 0 5 102 10 21 95 142 89 61 33 29",
    "L": "96 102 153 172 198 113 138 138 99 5 0 107 15 22 98 145 92 61 36 32",
    "K": "106 26 94 101 202 53 56 127 32 102 107 0 95 102 103 121 78 110 85 97",
    "M": "84 91 142 160 196 101 126 127 87 10 15 95 0 28 87 127 81 67 36 21",
    "F": "113 97 158 177 205 116 140 153 100 21 22 102 28 0 114 155 103 40 22 50",
    "P": "27 103 91 108 169 76 93 42 77 95 98 103 87 114 0 74 38 147 110 68",
    "S": "99 110 46 65 112 68 80 56 89 142 145 121 127 155 74 0 58 177 144 124",
    "T": "58 71 65 85 149 42 65 59 47 89 92 78 81 103 38 58 0 128 92 69",
    "W": "148 101 174 181 215 130 152 184 115 61 61 110 67 40 147 177 128 0 37 88",
    "Y": "112 77 143 160 194 99 122 147 83 33 36 85 36 22 110 144 92 37 0 55",
    "V": "64 96 133 152 192 96 121 109 84 29 32 97 21 50 68 124 69 88 55 0",
}
GRANTHAM = {aa: dict(zip(AMINO_ACIDS, map(int, row.split()))) for aa, row in _GRANTHAM_ROWS.items()}


def _numeric_summary(frame: pd.DataFrame, prefix: str, top_k: int = 3) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=frame.index)
    out[f"adv_{prefix}_mean"] = numeric.mean(axis=1)
    out[f"adv_{prefix}_median"] = numeric.median(axis=1)
    out[f"adv_{prefix}_std"] = numeric.std(axis=1)
    out[f"adv_{prefix}_min"] = numeric.min(axis=1)
    out[f"adv_{prefix}_max"] = numeric.max(axis=1)
    out[f"adv_{prefix}_skew_approx"] = (numeric.mean(axis=1) - numeric.median(axis=1)) / numeric.std(axis=1).replace(0, np.nan)
    out[f"adv_{prefix}_missing_count"] = numeric.isna().sum(axis=1).astype(float)
    out[f"adv_{prefix}_nonmissing_count"] = numeric.notna().sum(axis=1).astype(float)
    out[f"adv_{prefix}_zero_count"] = numeric.eq(0).sum(axis=1).astype(float)
    out[f"adv_{prefix}_positive_count"] = numeric.gt(0).sum(axis=1).astype(float)
    out[f"adv_{prefix}_negative_count"] = numeric.lt(0).sum(axis=1).astype(float)
    out[f"adv_{prefix}_high_risk_count"] = numeric.ge(numeric.quantile(0.75, axis=1), axis="index").sum(axis=1).astype(float)
    out[f"adv_{prefix}_low_risk_count"] = numeric.le(numeric.quantile(0.25, axis=1), axis="index").sum(axis=1).astype(float)
    sorted_values = np.sort(numeric.fillna(-np.inf).to_numpy(dtype=float), axis=1)
    finite = np.isfinite(sorted_values)
    top = np.where(finite[:, -top_k:], sorted_values[:, -top_k:], np.nan)
    bottom = np.where(finite[:, :top_k], sorted_values[:, :top_k], np.nan)
    out[f"adv_{prefix}_top{top_k}_mean"] = np.nanmean(top, axis=1)
    out[f"adv_{prefix}_top{top_k}_max"] = np.nanmax(top, axis=1)
    out[f"adv_{prefix}_bottom{top_k}_mean"] = np.nanmean(bottom, axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class AdvancedBioFeatureEngineer:
    """Stateless, label-free genomics feature layer for use after FeatureEngineer."""

    top_k: int = 3

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        def numeric_series(value: object) -> pd.Series:
            converted = pd.to_numeric(value, errors="coerce")
            if isinstance(converted, pd.Series):
                return converted.reindex(out.index).fillna(0.0)
            return pd.Series(float(converted) if pd.notna(converted) else 0.0, index=out.index)

        aa1 = out.get("AA_1", pd.Series("X", index=out.index)).astype(str).str.upper()
        aa2 = out.get("AA_2", pd.Series("X", index=out.index)).astype(str).str.upper()
        p1 = aa1.map(AA_PROPERTIES)
        p2 = aa2.map(AA_PROPERTIES)

        out["adv_blosum62"] = [BLOSUM62.get(a, {}).get(b, np.nan) for a, b in zip(aa1, aa2)]
        out["adv_grantham_distance"] = [GRANTHAM.get(a, {}).get(b, np.nan) for a, b in zip(aa1, aa2)]
        for prop in ("hydrophobicity", "polarity", "charge", "weight"):
            left = p1.map(lambda value: value.get(prop, np.nan) if isinstance(value, dict) else np.nan)
            right = p2.map(lambda value: value.get(prop, np.nan) if isinstance(value, dict) else np.nan)
            out[f"adv_{prop}_delta"] = right - left
            out[f"adv_{prop}_abs_delta"] = (right - left).abs()
        out["adv_aromaticity_change"] = (
            p2.map(lambda value: value.get("aromatic", 0) if isinstance(value, dict) else 0)
            - p1.map(lambda value: value.get("aromatic", 0) if isinstance(value, dict) else 0)
        )
        out["adv_sulfur_involvement"] = (aa1.isin(["C", "M"]) | aa2.isin(["C", "M"])).astype(float)
        out["adv_proline_impact"] = (aa1.eq("P") | aa2.eq("P")).astype(float)
        out["adv_glycine_impact"] = (aa1.eq("G") | aa2.eq("G")).astype(float)
        out["adv_cysteine_impact"] = (aa1.eq("C") | aa2.eq("C")).astype(float)
        cls1 = p1.map(lambda value: value.get("class", "unknown") if isinstance(value, dict) else "unknown")
        cls2 = p2.map(lambda value: value.get("class", "unknown") if isinstance(value, dict) else "unknown")
        out["adv_conservative_substitution"] = cls1.eq(cls2).astype(float)
        out["adv_radical_substitution"] = (out["adv_grantham_distance"] >= 100).astype(float)
        out["adv_aa_class_transition"] = (cls1 + "_to_" + cls2).map(AA_CLASS_TRANSITION_CODES).fillna(-1.0)
        out["adv_substitution_reversible"] = aa1.ne(aa2).astype(float)

        al_cols = [col for col in out.columns if col.startswith("AL_")]
        ek_cols = [col for col in out.columns if col.startswith("EK_")]
        if al_cols:
            out = pd.concat([out, _numeric_summary(out[al_cols], "al", self.top_k)], axis=1)
        if ek_cols:
            out = pd.concat([out, _numeric_summary(out[ek_cols], "ek", self.top_k)], axis=1)

        raw_af = [col for col in out.columns if col.startswith("AL_") and col.split("_")[-1].isdigit() and int(col.split("_")[-1]) <= 26]
        max_af = numeric_series(out.get("max_AF", out[raw_af].max(axis=1) if raw_af else 0.0))
        n_pops = numeric_series(out.get("n_pops", out[raw_af].notna().sum(axis=1) if raw_af else 0.0))
        path_count = numeric_series(out.get("n_EK_path", out.get("adv_ek_high_risk_count", 0.0)))
        benign_count = numeric_series(out.get("n_EK_ben", out.get("adv_ek_low_risk_count", 0.0)))
        ek_net = numeric_series(out.get("EK_net_evidence", path_count - benign_count))
        missing_burden = numeric_series(out.get("adv_al_missing_count", 0.0)) + numeric_series(out.get("adv_ek_missing_count", 0.0))
        evidence = np.c_[np.maximum(path_count, 0), np.maximum(benign_count, 0), np.maximum(n_pops / 5, 0)]
        evidence_total = evidence.sum(axis=1)
        proportions = np.divide(evidence, evidence_total[:, None], out=np.zeros_like(evidence), where=evidence_total[:, None] > 0)
        entropy = -(np.where(proportions > 0, proportions * np.log(proportions), 0)).sum(axis=1)

        out["adv_strong_benign_frequency_proxy"] = (max_af >= 0.05).astype(float)
        out["adv_moderate_benign_frequency_proxy"] = ((max_af >= 0.01) & (max_af < 0.05)).astype(float)
        out["adv_rarity_proxy"] = (max_af <= 1e-4).astype(float)
        out["adv_population_observed_count_proxy"] = n_pops
        out["adv_computational_pathogenic_count"] = path_count
        out["adv_computational_benign_count"] = benign_count
        out["adv_conflicting_evidence_count"] = np.minimum(path_count, benign_count)
        out["adv_missing_evidence_burden"] = missing_burden
        out["adv_evidence_balance_score"] = path_count - benign_count - 0.25 * n_pops
        out["adv_evidence_entropy_score"] = entropy

        conservation = numeric_series(out.get("adv_al_top3_mean", out.get("adv_al_mean", 0.0)))
        out["adv_max_af_x_conservation"] = max_af * conservation
        out["adv_rarity_x_pathogenic_evidence"] = out["adv_rarity_proxy"] * path_count
        out["adv_missingness_x_ek_net"] = missing_burden * ek_net
        out["adv_blosum_x_ek_risk"] = out["adv_blosum62"].fillna(0.0) * path_count
        out["adv_grantham_x_ek_risk"] = out["adv_grantham_distance"].fillna(0.0) * path_count
        out["adv_population_x_insilico"] = n_pops * ek_net
        out["adv_radical_x_conservation"] = out["adv_radical_substitution"] * conservation
        return out

    def feature_names(self, df: pd.DataFrame) -> list[str]:
        transformed = self.transform(df.head(1))
        return [column for column in transformed.columns if column.startswith("adv_")]
