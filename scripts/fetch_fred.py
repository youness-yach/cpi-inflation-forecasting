"""Download every FRED series the models use and save a monthly panel.

    python scripts/fetch_fred.py            # writes data/fred_monthly.csv

Uses FRED's public CSV download (fredgraph.csv), which needs no API key.
Daily series (oil, dollar index, credit spread) are reduced to the month-end
value, and quarterly GDP series are carried forward within the quarter, as in
the original notebook.
"""

import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

SERIES = {
    "CPI": "CPIAUCSL",                   # consumer price index (target is built from this)
    "FedFunds": "FEDFUNDS",              # policy rate
    "M2_nominal": "M2SL",                # money supply
    "GDP_real": "GDPC1",                 # real GDP (quarterly)
    "GDP_potential": "GDPPOT",           # CBO potential GDP (quarterly)
    "UnempRate": "UNRATE",               # unemployment rate
    "ExchangeRate": "DTWEXBGS",          # trade-weighted dollar (daily)
    "OilPrice": "DCOILWTICO",            # WTI crude (daily)
    "ImportPrices": "IR",                # import price index
    "FoodPPI": "WPU012",                 # PPI: farm products
    "RetailSales": "RSXFS",              # retail sales ex food services
    "WageGrowth": "CES0500000003",       # average hourly earnings
    "PPI_AllCommodities": "PPIACO",      # PPI: all commodities
    "PCE": "PCE",                        # personal consumption expenditures
    "CreditSpread": "BAA10Y",            # Baa corporate minus 10y Treasury (daily)
}
START = "2005-01-01"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
OUT = Path(__file__).resolve().parent.parent / "data" / "fred_monthly.csv"


def fetch(series_id: str) -> pd.Series:
    with urllib.request.urlopen(URL.format(sid=series_id, start=START), timeout=60) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df.columns = ["date", "value"]
    s = pd.to_numeric(df["value"], errors="coerce")   # FRED marks gaps with "."
    s.index = pd.to_datetime(df["date"])
    return s.dropna()


def main() -> None:
    frames = {}
    for name, sid in SERIES.items():
        print(f"  {name:20s} {sid}", end=" ", flush=True)
        s = fetch(sid)
        frames[name] = s.resample("ME").last()
        print(f"{len(s):>6} obs  {s.index.min().date()} → {s.index.max().date()}")
        time.sleep(0.5)
    panel = pd.DataFrame(frames)
    panel[["GDP_real", "GDP_potential"]] = panel[["GDP_real", "GDP_potential"]].ffill(limit=2)
    OUT.parent.mkdir(exist_ok=True)
    panel.to_csv(OUT, index_label="date")
    print(f"\nwrote {OUT}  ({len(panel)} months × {panel.shape[1]} series)")


if __name__ == "__main__":
    sys.exit(main())
