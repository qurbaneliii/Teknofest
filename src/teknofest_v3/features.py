from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


AA_PROPERTIES = {
    "A": (1.8, 89.09, 8.1, 0), "R": (-4.5, 174.2, 10.5, 1), "N": (-3.5, 132.12, 11.6, 0),
    "D": (-3.5, 133.10, 13.0, -1), "C": (2.5, 121.16, 5.5, 0), "Q": (-3.5, 146.15, 10.5, 0),
    "E": (-3.5, 147.13, 12.3, -1), "G": (-0.4, 75.07, 9.0, 0), "H": (-3.2, 155.16, 10.4, 1),
    "I": (4.5, 131.18, 5.2, 0), "L": (3.8, 131.18, 4.9, 0), "K": (-3.9, 146.19, 11.3, 1),
    "M": (1.9, 149.21, 5.7, 0), "F": (2.8, 165.19, 5.2, 0), "P": (-1.6, 115.13, 8.0, 0),
    "S": (-0.8, 105.09, 9.2, 0), "T": (-0.7, 119.12, 8.6, 0), "W": (-0.9, 204.23, 5.4, 0),
    "Y": (-1.3, 181.19, 6.2, 0), "V": (4.2, 117.15, 5.9, 0),
}


@dataclass
class V3FeatureBuilder:
    """Train-fitted, label-free feature builder; identifiers and labels never enter X."""

    feature_set: str = "v3_safe_minimal"
    numeric_columns_: list[str] = field(default_factory=list, init=False)
    categorical_columns_: list[str] = field(default_factory=list, init=False)
    category_frequency_: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def _eligible(self, frame: pd.DataFrame) -> list[str]:
        cols = [c for c in frame.columns if c not in {"Variant_ID", "Label"}]
        if self.feature_set == "v3_panel_robust":
            # A deliberately conservative, panel-agnostic subset selected only from training missingness.
            return [c for c in cols if c.startswith("EK_") or (c.startswith("AL_") and frame[c].isna().mean() <= 0.25) or c.startswith("CAT_") or c.startswith("AA_")]
        return cols

    def fit(self, train: pd.DataFrame) -> "V3FeatureBuilder":
        if "Label" in train.columns and any(c == "Label" for c in self._eligible(train)):
            raise AssertionError("Label entered eligible feature columns")
        selected = self._eligible(train)
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        for col in selected:
            converted = pd.to_numeric(train[col], errors="coerce")
            if pd.api.types.is_numeric_dtype(train[col]) or converted.notna().mean() >= 0.98:
                self.numeric_columns_.append(col)
            else:
                self.categorical_columns_.append(col)
        self.category_frequency_ = {
            col: (train[col].astype("string").fillna("__MISSING__").value_counts(normalize=True)).to_dict()
            for col in self.categorical_columns_
        }
        result = self._transform(train)
        self.feature_names_ = list(result.columns)
        self._assert_safe(result)
        return self

    def _frequency_features(self, source: pd.DataFrame, result: pd.DataFrame) -> None:
        al_cols = [c for c in self.numeric_columns_ if c.startswith("AL_")]
        ek_cols = [c for c in self.numeric_columns_ if c.startswith("EK_")]
        if al_cols:
            al = source[al_cols].apply(pd.to_numeric, errors="coerce")
            clipped = al.clip(lower=0)
            result["v3_al_mean"] = al.mean(axis=1)
            result["v3_al_max"] = al.max(axis=1)
            result["v3_al_nonmissing"] = al.notna().sum(axis=1)
            result["v3_al_rare_count"] = clipped.le(1e-4).sum(axis=1)
            result["v3_al_high_frequency_count"] = clipped.ge(0.01).sum(axis=1)
            if self.feature_set in {"v3_frequency_heavy", "v3_bio_full"}:
                result["v3_al_log1p_mean"] = np.log1p(clipped).mean(axis=1)
                result["v3_al_log1p_max"] = np.log1p(clipped).max(axis=1)
        if ek_cols:
            ek = source[ek_cols].apply(pd.to_numeric, errors="coerce")
            result["v3_ek_mean"] = ek.mean(axis=1)
            result["v3_ek_std"] = ek.std(axis=1)
            result["v3_ek_nonmissing"] = ek.notna().sum(axis=1)
            if al_cols:
                result["v3_rarity_x_ek"] = result["v3_al_rare_count"] * result["v3_ek_mean"]

    def _aa_features(self, source: pd.DataFrame, result: pd.DataFrame) -> None:
        left = source.get("AA_1", pd.Series("X", index=source.index)).astype("string").str.upper().str[0]
        right = source.get("AA_2", pd.Series("X", index=source.index)).astype("string").str.upper().str[0]
        for name, pos in (("hydrophobicity", 0), ("weight", 1), ("polarity", 2), ("charge", 3)):
            a = left.map(lambda aa: AA_PROPERTIES.get(aa, (np.nan,) * 4)[pos])
            b = right.map(lambda aa: AA_PROPERTIES.get(aa, (np.nan,) * 4)[pos])
            result[f"v3_aa_{name}_delta"] = b - a
            result[f"v3_aa_{name}_abs_delta"] = (b - a).abs()
        result["v3_aa_unknown"] = (~left.isin(AA_PROPERTIES) | ~right.isin(AA_PROPERTIES)).astype(float)
        result["v3_aa_proline_involved"] = (left.eq("P") | right.eq("P")).astype(float)
        result["v3_aa_glycine_involved"] = (left.eq("G") | right.eq("G")).astype(float)
        result["v3_aa_cysteine_involved"] = (left.eq("C") | right.eq("C")).astype(float)

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for col in self.numeric_columns_:
            result[col] = pd.to_numeric(frame.get(col), errors="coerce")
        for col in self.categorical_columns_:
            values = frame.get(col, pd.Series("__MISSING__", index=frame.index)).astype("string").fillna("__MISSING__")
            result[f"v3_freq_{col}"] = values.map(self.category_frequency_[col]).fillna(0.0)
        result["v3_global_missing_rate"] = result.isna().mean(axis=1)
        if self.feature_set in {"v3_frequency_heavy", "v3_bio_full", "v3_panel_robust"}:
            self._frequency_features(frame, result)
        if self.feature_set in {"v3_aa_heavy", "v3_bio_full"}:
            self._aa_features(frame, result)
        return result.replace([np.inf, -np.inf], np.nan).astype(float)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("V3FeatureBuilder must be fitted on training data before transform.")
        result = self._transform(frame).reindex(columns=self.feature_names_)
        self._assert_safe(result)
        return result

    @staticmethod
    def _assert_safe(features: pd.DataFrame) -> None:
        forbidden = [c for c in features.columns if c.lower() in {"variant_id", "label"} or "variant_id" in c.lower() or c.lower().startswith("label")]
        if forbidden:
            raise AssertionError(f"Forbidden feature names: {forbidden}")
        if not all(pd.api.types.is_numeric_dtype(features[c]) for c in features):
            raise AssertionError("V3 features must be numeric only")

    def save_schema(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"feature_set": self.feature_set, "features": self.feature_names_}, indent=2), encoding="utf-8")
