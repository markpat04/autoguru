"""Shared prediction logic used by both the FastAPI app and the Gradio UI.

Loads the trained AutoGluon predictor plus the artifacts produced by
train.py (feature importance, prediction interval, feature value lists).
"""
import json
import os

import pandas as pd
from autogluon.tabular import TabularPredictor

PROJECT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT, "models", "autogluon")
ARTIFACTS = os.path.join(PROJECT, "artifacts")

FEATURES = [
    "brand", "model", "model_year", "milage", "fuel_type", "engine",
    "transmission", "ext_col", "int_col", "accident", "clean_title",
]
LABEL = "price"

_predictor = None
_importance = None
_intervals = None
_feature_values = None


def get_predictor() -> TabularPredictor:
    global _predictor
    if _predictor is None:
        _predictor = TabularPredictor.load(MODEL_PATH)
    return _predictor


def _load(name, loader):
    path = os.path.join(ARTIFACTS, name)
    with open(path) as f:
        return loader(f)


def get_importance() -> pd.DataFrame:
    global _importance
    if _importance is None:
        _importance = pd.read_csv(os.path.join(ARTIFACTS, "feature_importance.csv"),
                                  index_col=0)
    return _importance


def get_intervals() -> dict:
    global _intervals
    if _intervals is None:
        _intervals = _load("intervals.json", json.load)
    return _intervals


def get_feature_values() -> dict:
    global _feature_values
    if _feature_values is None:
        _feature_values = _load("feature_values.json", json.load)
    return _feature_values


def predict_car(features: dict) -> dict:
    """Predict price for one car. `features` keys must match FEATURES."""
    row = {k: features[k] for k in FEATURES}  # KeyError on missing -> 422 upstream
    df = pd.DataFrame([row])
    price = float(get_predictor().predict(df).iloc[0])
    half = get_intervals()["half_width"]

    top = []
    for feat, imp_row in get_importance().head(5).iterrows():
        top.append({
            "feature": feat,
            "value": str(row[feat]),
            "importance": round(float(imp_row["importance"]), 4),
        })

    return {
        "price": round(price, 2),
        "interval_low": round(max(0.0, price - half), 2),
        "interval_high": round(price + half, 2),
        "top_factors": top,
    }
