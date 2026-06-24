"""Shared numeric preprocessing contract for V3 model families.

The evaluator keeps the fitted imputer/scaler inside each model pipeline so
local_test and panel rows are transformed only by train-fitted state.
"""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def scaled_numeric_pipeline(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])
