"""Evaluate AutoGuru on the frozen holdout test set.

Prints: RMSE, MAE, R^2 + p95 (and mean) single-prediction latency.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from predictor_service import LABEL, get_predictor

PROJECT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    test_df = pd.read_parquet(os.path.join(PROJECT, "artifacts", "test.parquet"))
    X = test_df.drop(columns=[LABEL])
    y = test_df[LABEL].to_numpy()

    predictor = get_predictor()
    t0 = time.perf_counter()
    y_pred = predictor.predict(X).to_numpy()
    bulk_s = time.perf_counter() - t0

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))

    # single-row latency (what /predict does per request)
    lat = []
    for i in range(min(200, len(X))):
        s = time.perf_counter()
        predictor.predict(X.iloc[[i]])
        lat.append((time.perf_counter() - s) * 1000)
    lat = np.array(lat)

    print(f"test rows            : {len(test_df)}")
    print(f"RMSE                 : ${rmse:,.0f}")
    print(f"MAE                  : ${mae:,.0f}")
    print(f"R^2                  : {r2:.4f}")
    print(f"mean price (test)    : ${y.mean():,.0f}")
    print(f"bulk predict         : {bulk_s:.2f}s for {len(X)} rows")
    print(f"single-row latency   : mean {lat.mean():.1f} ms | p95 {np.quantile(lat, 0.95):.1f} ms")


if __name__ == "__main__":
    main()
