"""Naive benchmarks for the v2 test window (Jan 2023 - Dec 2025), for context on the model R².

    python scripts/baselines.py
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

d = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "fred_monthly.csv",
                index_col=0, parse_dates=True).loc["2005":"2025-12"].ffill()
mom = d["CPI"].pct_change() * 100
test, train = mom["2023-01":"2025-12"], mom["2008-01":"2022-12"]
benchmarks = {
    "Train mean (2008-2022)": pd.Series(train.mean(), index=test.index),
    "Last month (random walk)": mom.shift(1)[test.index],
    "Trailing 12-month mean": mom.rolling(12).mean().shift(1)[test.index],
}
for name, pred in benchmarks.items():
    print(f"{name:26s}  MSE {mean_squared_error(test, pred):.6f}   R² {r2_score(test, pred):+.3f}")
