"""README figures: where v1's skill came from, and how v2 compares with naive benchmarks.

    python scripts/make_figures.py

Feature engineering is not re-implemented here. The definition cells are
executed straight from notebooks/v2_next_month.ipynb (identical in v1), so
these charts use exactly the notebook's features, split and Lasso setup.
Also copies the notebooks' own charts into reports/figures/.
"""

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
NB = ROOT / "notebooks"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
V1, V2, BENCH = "#898781", "#2a78d6", "#c3c2b7"      # v1 = muted (superseded), v2 = series blue
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False})

# same drop list as the notebooks (non-stationary levels)
DROP = ['CPI', 'FedFunds', 'M2_nominal', 'GDP_real', 'GDP_potential', 'UnempRate', 'ExchangeRate',
        'ImportPrices', 'FoodPPI', 'RetailSales', 'WageGrowth', 'OilPrice', 'PPI_AllCommodities', 'PCE',
        'CreditSpread', 'M2_real', 'Oil_lag2', 'Oil_lag3', 'PPI_lag2', 'PPI_lag3', 'Import_lag3',
        'FedFunds_lag3', 'FedFunds_lag6', 'CPI_lag1', 'CPI_lag2', 'CPI_lag3', 'CPI_lag6', 'CPI_lag12']


def notebook_defs() -> dict:
    ns = {}
    cells = json.loads((NB / "v2_next_month.ipynb").read_text())["cells"]
    for c in cells:
        s = "".join(c["source"])
        if c["cell_type"] == "code" and any(k in s for k in ("import pandas", "START_DATE =", "def load_fred",
                                                               "def compute_derived_features", "def clean_and_split")):
            exec(s, ns)
    return ns


def dataset(ns: dict, lag: bool):
    raw = ns["load_fred"](str(ROOT / "data" / "fred_monthly.csv"), ns["START_DATE"], ns["END_DATE"])
    f = ns["compute_derived_features"](raw).drop(DROP, axis=1)
    if lag:
        target = f["CPI_MoM"]
        f = f.drop(columns=["CPI_MoM"]).shift(1)
        f.insert(0, "CPI_MoM", target)
    return ns["clean_and_split"](f, ns["TRAIN_END"], ns["TEST_START"])


def fit(train, test):
    Xtr, Xte = train.drop(columns=["CPI_MoM"]), test.drop(columns=["CPI_MoM"])
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    ytr, yte = train["CPI_MoM"], test["CPI_MoM"]
    tscv = TimeSeriesSplit(n_splits=5)
    lasso = LassoCV(alphas=np.logspace(-6, 0, 100), cv=tscv, max_iter=10000).fit(Xtr_s, ytr)
    # Ridge alpha chosen exactly as in the notebook (mean TimeSeriesSplit R² over a grid)
    alphas = np.logspace(-4, 4, 200)
    from sklearn.metrics import r2_score
    scores = [np.mean([r2_score(ytr.iloc[v], Ridge(alpha=a).fit(Xtr_s[t], ytr.iloc[t]).predict(Xtr_s[v]))
                       for t, v in tscv.split(Xtr_s)]) for a in alphas]
    ridge = Ridge(alpha=alphas[int(np.argmax(scores))]).fit(Xtr_s, ytr)
    ols = LinearRegression().fit(Xtr_s, ytr)
    mse = {k: mean_squared_error(yte, m.predict(Xte_s)) for k, m in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]}
    coefs = pd.Series(lasso.coef_, index=Xtr.columns)
    return mse, coefs


def title(ax, head, sub):
    ax.set_title(head, loc="left", fontsize=13, fontweight="bold", pad=24)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def correlations(ns):
    full1, _, _ = dataset(ns, lag=False)
    full2, _, _ = dataset(ns, lag=True)
    c1 = full1.corr()["CPI_MoM"].drop(["CPI_MoM"])
    c2 = full2.corr()["CPI_MoM"].drop(["CPI_MoM"])
    top = c1.abs().sort_values(ascending=False).head(10).index[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(y + 0.2, c1[top], height=0.38, color=V1, label="Same month (v1)")
    ax.barh(y - 0.2, c2[top], height=0.38, color=V2, label="Previous month (v2)")
    for i, f in enumerate(top):
        ax.text(max(c1[f], 0) + 0.015, i + 0.2, f"{c1[f]:+.2f}", va="center", fontsize=8.5, color=INK2)
        ax.text(max(c2[f], 0) + 0.015, i - 0.2, f"{c2[f]:+.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, top)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-0.55, 0.95)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(length=0)
    ax.legend(loc="lower right", frameon=False)
    title(ax, "Where v1's accuracy came from",
          "Correlation with the target (CPI month-over-month), 2008–2025: the 10 strongest v1 features")
    fig.tight_layout()
    fig.savefig(FIG / "leakage_correlations.png", dpi=150)


def mse_chart(mse1, mse2, bench):
    rows = [(f"{k} · v1 (same month)", v, V1) for k, v in mse1.items()] + \
           [(f"{k} · v2 (previous month)", v, V2) for k, v in mse2.items()] + \
           [(k, v, BENCH) for k, v in bench.items()]
    rows.sort(key=lambda r: r[1])
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.0008, i, f"{r[1]:.4f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(length=0)
    ax.set_xlabel("Test MSE, Jan 2023 – Dec 2025 (lower is better)")
    ax.legend(handles=[Patch(color=V1, label="v1: uses same-month data"), Patch(color=V2, label="v2: honest forecast"),
                       Patch(color=BENCH, label="naive benchmark")], loc="upper right", frameon=False)
    title(ax, "Honest forecasts vs. simple rules",
          "With only past data, Lasso lands among the naive benchmarks")
    fig.tight_layout()
    fig.savefig(FIG / "mse_vs_benchmarks.png", dpi=150)


def coef_chart(k1, k2):
    keep = k1.index[(k1 != 0) | (k2 != 0)]
    order = (k1[keep].abs() + k2[keep].abs()).sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.barh(y + 0.2, k1[order], height=0.38, color=V1, label=f"v1: {int((k1 != 0).sum())} features kept")
    ax.barh(y - 0.2, k2[order], height=0.38, color=V2, label=f"v2: {int((k2 != 0).sum())} features kept")
    ax.set_yticks(y, order)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(length=0)
    ax.set_xlabel("Standardised Lasso coefficient")
    ax.legend(loc="lower right", frameon=False)
    title(ax, "What Lasso relies on, before and after the fix",
          "Coefficients on standardised features; features zeroed in both versions omitted")
    fig.tight_layout()
    fig.savefig(FIG / "lasso_coefficients.png", dpi=150)


def copy_notebook_charts():
    want = {"#plot predictions": "scatter", "Backtest: Predicted vs Actual": "backtest",
            "Bootstrap Uncertainty": "bootstrap"}
    for nb, tag in [("v1_original", "v1"), ("v2_next_month", "v2")]:
        for c in json.loads((NB / f"{nb}.ipynb").read_text())["cells"]:
            s = "".join(c["source"])
            imgs = [o["data"]["image/png"] for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
            for key, name in want.items():
                if imgs and key in s:
                    (FIG / f"{tag}_{name}.png").write_bytes(base64.b64decode(imgs[0]))


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    ns = notebook_defs()
    _, tr1, te1 = dataset(ns, lag=False)
    _, tr2, te2 = dataset(ns, lag=True)
    mse1, k1 = fit(tr1, te1)
    mse2, k2 = fit(tr2, te2)
    mom = te2["CPI_MoM"]
    raw = ns["load_fred"](str(ROOT / "data" / "fred_monthly.csv"), ns["START_DATE"], ns["END_DATE"])
    m = raw["CPI"].pct_change() * 100
    bench = {"Train mean (2008–2022)": mean_squared_error(mom, np.full(len(mom), m["2008-01":"2022-12"].mean())),
             "Trailing 12-month mean": mean_squared_error(mom, m.rolling(12).mean().shift(1)[mom.index]),
             "Last month (random walk)": mean_squared_error(mom, m.shift(1)[mom.index])}
    print("v1", {k: round(v, 6) for k, v in mse1.items()}, "\nv2", {k: round(v, 6) for k, v in mse2.items()})
    print("lasso non-zero v1/v2:", int((k1 != 0).sum()), int((k2 != 0).sum()))
    print("top v2 coefs:\n", k2[k2 != 0].sort_values(key=abs, ascending=False).round(4).to_string())
    correlations(ns)
    mse_chart(mse1, mse2, bench)
    coef_chart(k1, k2)
    copy_notebook_charts()
    print("figures written to", FIG)
