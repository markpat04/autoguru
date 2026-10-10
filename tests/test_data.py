"""Data validation tests for the training dataset."""
import os

import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT, "data", "used_cars.parquet")

REQUIRED_COLUMNS = {
    "brand", "model", "model_year", "milage", "fuel_type", "engine",
    "transmission", "ext_col", "int_col", "accident", "clean_title", "price",
}


def test_dataset_exists_and_has_required_columns():
    assert os.path.exists(DATA_PATH), "dataset parquet missing"
    df = pd.read_parquet(DATA_PATH)
    assert REQUIRED_COLUMNS.issubset(set(df.columns)), (
        f"missing columns: {REQUIRED_COLUMNS - set(df.columns)}"
    )


def test_label_is_valid():
    df = pd.read_parquet(DATA_PATH)
    assert df["price"].notna().all(), "null prices found"
    assert (df["price"] > 0).all(), "non-positive prices found"
    assert len(df) >= 5000, f"dataset unexpectedly small: {len(df)} rows"


def test_core_features_mostly_present():
    df = pd.read_parquet(DATA_PATH)
    for col in ("model_year", "milage"):
        null_frac = df[col].isna().mean()
        assert null_frac < 0.05, f"{col} has {null_frac:.1%} nulls"
