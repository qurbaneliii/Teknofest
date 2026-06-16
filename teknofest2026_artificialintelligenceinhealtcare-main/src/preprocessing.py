from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


@dataclass
class FeatureTypes:
    categorical: List[str]
    numerical: List[str]


def _safe_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # Compatibility with older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def identify_feature_types(
    df: pd.DataFrame,
    target: str,
    categorical_max_cardinality: int = 20,
) -> FeatureTypes:
    feature_cols = [c for c in df.columns if c != target]
    categorical: List[str] = []
    numerical: List[str] = []

    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            nunique = series.nunique(dropna=True)
            if nunique <= categorical_max_cardinality:
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            categorical.append(col)

    return FeatureTypes(categorical=categorical, numerical=numerical)


class SklearnPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target: str,
        categorical_max_cardinality: int = 20,
        drop_columns: Optional[List[str]] = None,
    ) -> None:
        self.target = target
        self.categorical_max_cardinality = categorical_max_cardinality
        self.drop_columns = drop_columns or []
        self.feature_types_: Optional[FeatureTypes] = None
        self.column_transformer_: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SklearnPreprocessor":
        X = X.copy()
        if self.drop_columns:
            X = X.drop(columns=[c for c in self.drop_columns if c in X.columns])

        self.feature_types_ = identify_feature_types(
            X, target=self.target, categorical_max_cardinality=self.categorical_max_cardinality
        )

        num_pipe = SimpleImputer(strategy="median", add_indicator=True)
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _safe_one_hot_encoder()),
            ]
        )

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.feature_types_.numerical),
                ("cat", cat_pipe, self.feature_types_.categorical),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        self.column_transformer_.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        if self.column_transformer_ is None:
            raise RuntimeError("Preprocessor is not fitted")

        X = X.copy()
        if self.drop_columns:
            X = X.drop(columns=[c for c in self.drop_columns if c in X.columns])

        return self.column_transformer_.transform(X)

    def get_feature_names_out(self) -> List[str]:
        if self.column_transformer_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        try:
            return list(self.column_transformer_.get_feature_names_out())
        except Exception:
            # Fallback for older sklearn
            names: List[str] = []
            if self.feature_types_:
                names.extend(self.feature_types_.numerical)
                names.extend(self.feature_types_.categorical)
            return names
