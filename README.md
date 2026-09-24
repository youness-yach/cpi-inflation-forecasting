# Forecasting Monthly US Inflation with Regularised Regression

Can macro data from FRED predict next month's CPI change? OLS, Ridge and Lasso
are trained on 28 engineered features covering monetary policy, cost-push,
activity and demand, from 2008 to 2022, and tested on 2023–2025. Baruch
College Pre-MFE, Machine Learning.

The project exists in two versions, and the difference between them is the
point.

| | v1: original submission | v2: next-month forecast |
|---|---:|---:|
| Features measured in | the **same** month as the target | the **previous** month |
| Lasso test R² | 0.394 | **−0.199** |
| Ridge test R² | 0.311 | −0.504 |
| OLS test R² | 0.051 | −2.463 |
| Lasso test MSE | 0.0106 | 0.0210 |

(v1 figures are re-run on the current FRED vintage. The original submission
reported 0.388 / 0.301 / 0.049.)

## What changed and why

In v1, every feature is measured in the same month as the target, month-over-month
CPI change. Two of them give the answer away:

- `CPI_YoY` is computed from that month's CPI.
- `PPI_MoM`, `Oil_MoM` and `FoodPPI_MoM` are same-month figures, published
  about when CPI itself is. `PPI_MoM` alone correlates 0.77 with the target.

So v1 is a same-month *nowcast* with look-ahead, not a forecast.

**v2 changes one thing.** Every feature is lagged one month, so month *t* is
predicted only from what was known by the end of month *t − 1*:

```python
target = df_feat["CPI_MoM"]
df_feat = df_feat.drop(columns=["CPI_MoM"]).shift(1)
df_feat.insert(0, "CPI_MoM", target)
```

Everything else is identical: features, collinearity filter, scaling, OLS,
Ridge and Lasso, 5-fold `TimeSeriesSplit` for α, the 2008–2022 / 2023–2025
split, and 10,000-draw bootstrap bands.

![v2 backtest](reports/figures/v2_backtest.png)

## Reading the result

A negative R² means the model does worse than predicting the test period's
average. For context, naive benchmarks on the same window
([`scripts/baselines.py`](scripts/baselines.py)):

| Benchmark | Test MSE | R² |
|---|---:|---:|
| Train mean (2008–2022) | 0.0199 | −0.140 |
| **Lasso (v2)** | **0.0210** | **−0.199** |
| Trailing 12-month mean | 0.0212 | −0.212 |
| Last month (random walk) | 0.0294 | −0.684 |

Lasso lands among the naive benchmarks: better than a random walk, no better
than a historical average. That matches a long line of findings that monthly
inflation is hard to forecast beyond simple benchmarks. Lasso still ranks
first of the three models, and regularisation helps a lot (OLS overfits to
R² −2.46), but the macro features carry little next-month signal. Most of v1's
apparent skill came from same-month producer prices.

<details><summary>Bootstrap uncertainty bands (v2)</summary>

![v2 bootstrap](reports/figures/v2_bootstrap.png)
</details>

## Run it

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
python scripts/fetch_fred.py          # data/fred_monthly.csv, FRED public CSV, no API key
jupyter nbconvert --to notebook --execute --inplace notebooks/v2_next_month.ipynb
python scripts/baselines.py
```

The committed `data/fred_monthly.csv` is the September 2026 vintage. FRED
revises history, so a fresh download can shift results in the third decimal.

## Notes

- **v1 edits.** v1 runs from the saved CSV instead of the FRED API, has one
  undefined variable removed so it runs in a fresh kernel, and prints "MSE"
  where the original said "RMSE". The modelling is untouched.
- **Missing October 2025.** October 2025 CPI was never published (US
  government shutdown). As in the original pipeline, it is forward-filled.
  That makes October's change 0% and folds two months into November.
- **Unused feature sets.** The collinearity-filtered and correlation-selected
  feature sets are built in both notebooks, but the final models use the full
  scaled set, as in the original.

## Layout

```
├── notebooks/  v1_original.ipynb · v2_next_month.ipynb
├── scripts/    fetch_fred.py · baselines.py
├── data/       fred_monthly.csv (15 FRED series, monthly)
└── reports/figures/
```

---
Youness Yachruti · [LinkedIn](https://www.linkedin.com/in/youness-yachruti/)
