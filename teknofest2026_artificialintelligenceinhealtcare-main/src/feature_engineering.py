from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class GroupFeatureSpec:
    prefix: str
    numeric_aggs: List[str]
    add_missing_count: bool = True


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        group_specs: Optional[List[GroupFeatureSpec]] = None,
        drop_high_missing: bool = True,
        high_missing_threshold: float = 0.95,
    ) -> None:
        if group_specs is None:
            group_specs = [
                GroupFeatureSpec(prefix="AL", numeric_aggs=["mean", "std", "min", "max"]),
                GroupFeatureSpec(prefix="EK", numeric_aggs=["mean", "std", "min", "max"]),
                GroupFeatureSpec(prefix="AA", numeric_aggs=["mean", "std", "min", "max"]),
                GroupFeatureSpec(prefix="CAT", numeric_aggs=["mean", "std", "min", "max"]),
            ]
        self.group_specs = group_specs
        self.drop_high_missing = drop_high_missing
        self.high_missing_threshold = high_missing_threshold
        self.high_missing_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        if self.drop_high_missing:
            missing_pct = X.isna().mean()
            self.high_missing_columns_ = missing_pct[missing_pct >= self.high_missing_threshold].index.tolist()
        else:
            self.high_missing_columns_ = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.high_missing_columns_:
            X = X.drop(columns=[c for c in self.high_missing_columns_ if c in X.columns])

        for spec in self.group_specs:
            cols = [c for c in X.columns if c.startswith(f"{spec.prefix}_")]
            if not cols:
                continue

            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
            if num_cols:
                row_values = X[num_cols]
                if "mean" in spec.numeric_aggs:
                    X[f"{spec.prefix}_mean"] = row_values.mean(axis=1)
                if "std" in spec.numeric_aggs:
                    X[f"{spec.prefix}_std"] = row_values.std(axis=1)
                if "min" in spec.numeric_aggs:
                    X[f"{spec.prefix}_min"] = row_values.min(axis=1)
                if "max" in spec.numeric_aggs:
                    X[f"{spec.prefix}_max"] = row_values.max(axis=1)

            if spec.add_missing_count:
                X[f"{spec.prefix}_missing"] = X[cols].isna().sum(axis=1)

        # Global missing count
        X["missing_total"] = X.isna().sum(axis=1)

        # Numeric stability
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        return X
