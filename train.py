"""Train AutoGuru's used-car price model.

Steps:
  1. Load data/data/used_cars.parquet (public HF dataset, see data/README.md)
  2. Minimal cleaning (drop rows missing target / core numerics)
  3. Deterministic 70/15/15 train/val/test split (seed=42)
  4. Train AutoGluon TabularPredictor (LightGBM + XGBoost, CPU only)
  5. Save leaderboard, permutation feature importance, empirical
     prediction intervals, and UI feature values under artifacts/
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor

SEED = 42
LABEL = "price"
PROJECT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT, "data", "used_cars.parquet")
ARTIFACTS = os.path.join(PROJECT, "artifacts")
MODEL_PATH = os.path.join(PROJECT, "models", "autogluon")

CATEGORICALS = [
    "brand", "model", "fuel_type", "engine", "transmission",
    "ext_col", "int_col", "accident", "clean_title",
]
NUMERICS = ["model_year", "milage"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    before = len(df)
    df = df.dropna(subset=[LABEL, "model_year", "milage"]).reset_index(drop=True)
    # Price sanity filter: drop obvious data-entry errors. The raw data
    # contains listings like a 2005 Maserati at $2,954,083 (a ~$20k car).
    # Only rows with price > $500k are removed (3 of 7970); documented here
    # and in the README rather than silently dropped.
    n_price = len(df)
    df = df[df[LABEL] <= 500_000].reset_index(drop=True)
    print(f"cleaned: {before} -> {len(df)} rows "
          f"(dropped {before - n_price} nulls, {n_price - len(df)} price outliers)")
    return df


def split(df: pd.DataFrame):
    train_df, temp = train_test_split(df, test_size=0.30, random_state=SEED)
    val_df, test_df = train_test_split(temp, test_size=0.50, random_state=SEED)
    print(f"split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def build_feature_values(train_df: pd.DataFrame) -> dict:
    """Value lists / defaults for the Gradio form, derived from training data."""
    fv = {}
    top_n = {"brand": 40, "model": 60, "ext_col": 20, "int_col": 20}
    for col in CATEGORICALS:
        n = top_n.get(col, 999)
        fv[col] = train_df[col].value_counts().head(n).index.astype(str).tolist()
    fv["defaults"] = {
        "model_year": float(train_df["model_year"].median()),
        "milage": float(train_df["milage"].median()),
        **{c: str(train_df[c].mode().iloc[0]) for c in CATEGORICALS},
    }
    return fv


def main() -> None:
    os.makedirs(ARTIFACTS, exist_ok=True)
    df = load_and_clean()
    train_df, val_df, test_df = split(df)

    predictor = TabularPredictor(
        label=LABEL,
        path=MODEL_PATH,
        eval_metric="root_mean_squared_error",
        verbosity=2,
    )
    predictor.fit(
        train_df,
        time_limit=300,
        presets="medium_quality",
        hyperparameters={"GBM": {}, "XGB": {}},
    )

    # 1) leaderboard (scored on the frozen test set for honest reporting)
    lb = predictor.leaderboard(data=test_df, silent=True)
    lb.to_csv(os.path.join(ARTIFACTS, "leaderboard.csv"), index=False)
    cols = [c for c in ["model", "score_test", "score_val",
                        "pred_time_test", "fit_time"] if c in lb.columns]
    print(lb[cols].to_string())

    # 2) permutation feature importance on a validation sample
    fi = predictor.feature_importance(
        data=val_df.sample(min(1000, len(val_df)), random_state=SEED),
        num_shuffle_sets=5,
        silent=True,
    )
    fi.to_csv(os.path.join(ARTIFACTS, "feature_importance.csv"))
    print(fi.head(8).to_string())

    # 3) empirical 90% prediction interval from validation residuals
    val_pred = predictor.predict(val_df).to_numpy()
    half_width = float(np.quantile(np.abs(val_df[LABEL].to_numpy() - val_pred), 0.90))
    with open(os.path.join(ARTIFACTS, "intervals.json"), "w") as f:
        json.dump(
            {
                "half_width": half_width,
                "level": 0.90,
                "method": "empirical quantile of |residual| on validation set "
                          "(symmetric interval; documented limitation)",
            },
            f,
            indent=2,
        )
    print(f"90% interval half-width: ${half_width:,.0f}")

    # 4) feature values + frozen splits for eval/tests
    with open(os.path.join(ARTIFACTS, "feature_values.json"), "w") as f:
        json.dump(build_feature_values(train_df), f, indent=2)
    val_df.to_parquet(os.path.join(ARTIFACTS, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(ARTIFACTS, "test.parquet"), index=False)
    print("artifacts written.")


if __name__ == "__main__":
    main()
