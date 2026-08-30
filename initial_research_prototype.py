"""Initial notebook-style prototype for the MSCI ACWI TSMOM project.

This was the first implementation used to understand the mechanics of the
strategy: download the data, calculate one 252-day momentum signal, apply an
execution lag, combine equity and cash returns, and inspect the resulting
wealth path.  It is intentionally linear and tests only one specification.

The portfolio results in this repository come from ``tsmom_backtest.py``.  This
prototype is retained to show how the research process developed.
"""

# Research plan
# 1. Load ACWI gross and net total return data.
# 2. Load the three-month Treasury yield.
# 3. Calculate equity and cash returns.
# 4. Calculate 252-day absolute momentum.
# 5. Convert momentum into a long/cash signal.
# 6. Apply the conservative execution lag.
# 7. Calculate strategy returns, costs, and wealth.
# 8. Calculate basic performance statistics.


# %% Imports and assumptions

from datetime import date
from io import StringIO
from pathlib import Path

import certifi
import matplotlib.pyplot as plt
import pandas as pd
import requests
from mscidata import msci


START_DATE = "1987-01-01"
END_DATE = date.today().isoformat()
LOOKBACK = 252
TRANSACTION_COST = 0.0005
ANNUAL_ETF_EXPENSE = 0.0032
LOCAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


# %% Download MSCI ACWI gross and net total return levels

# Gross levels generate the signal. Net levels represent realized equity
# returns after dividend withholding taxes.
gross = msci.get_levels(
    "892400",
    START_DATE,
    END_DATE,
    variant="gross",
)

net = msci.get_levels(
    "892400",
    START_DATE,
    END_DATE,
    variant="net",
)


# %% Download the US three-month Treasury yield from FRED

fred_url = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    f"?id=DGS3MO&cosd={START_DATE}&coed={END_DATE}"
)
response = requests.get(fred_url, timeout=60, verify=certifi.where())
response.raise_for_status()
risk_free = pd.read_csv(StringIO(response.text))


# %% Save local source snapshots for the main parameter study

# These files are intentionally ignored by Git. Use the MSCI download only
# when your access and license authorize it.
download_date = date.today().strftime("%Y%m%d")
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
gross.to_csv(
    LOCAL_DATA_DIR / f"msci_acwi_grtr_{download_date}.csv",
    index=False,
)
net.to_csv(
    LOCAL_DATA_DIR / f"msci_acwi_netr_{download_date}.csv",
    index=False,
)
risk_free.to_csv(
    LOCAL_DATA_DIR / f"fred_dgs3mo_{download_date}.csv",
    index=False,
)
print(f"Saved local source snapshots in {LOCAL_DATA_DIR}")


# %% Clean the three source datasets

gross = gross[["DATE", "LEVEL"]].rename(
    columns={"DATE": "date", "LEVEL": "gross_level"}
)
net = net[["DATE", "LEVEL"]].rename(
    columns={"DATE": "date", "LEVEL": "net_level"}
)

gross["date"] = pd.to_datetime(gross["date"])
net["date"] = pd.to_datetime(net["date"])
gross = gross.sort_values("date")
net = net.sort_values("date")

risk_free = risk_free.rename(
    columns={"observation_date": "date", "DGS3MO": "annual_yield"}
)
risk_free["date"] = pd.to_datetime(risk_free["date"])
risk_free["annual_yield"] = pd.to_numeric(
    risk_free["annual_yield"].replace(".", pd.NA), errors="coerce"
)
risk_free = risk_free.dropna(subset=["annual_yield"]).sort_values("date")


# %% Align the datasets and calculate returns

market = gross.merge(net, on="date", how="inner", validate="one_to_one")

# Backward alignment prevents a future Treasury observation from being used.
market = pd.merge_asof(
    market.sort_values("date"),
    risk_free,
    on="date",
    direction="backward",
).dropna(subset=["annual_yield"])

market["calendar_days"] = market["date"].diff().dt.days.fillna(0)
market["net_return"] = market["net_level"].pct_change(fill_method=None).fillna(0.0)

# The yield known at the preceding close accrues over the next interval.
prior_yield = market["annual_yield"].shift(1).ffill()
prior_yield.iloc[0] = market["annual_yield"].iloc[0]
market["cash_return"] = (
    (1.0 + prior_yield / 100.0) ** (market["calendar_days"] / 365.2425) - 1.0
)


# %% Calculate 252-day absolute momentum and the long/cash position

market["momentum_252"] = market["gross_level"].pct_change(
    periods=LOOKBACK,
    fill_method=None,
)
market["signal"] = (
    (market["momentum_252"] > 0)
    .astype(float)
    .where(market["momentum_252"].notna())
)

# Signal at close t -> trade at close t+1 -> earn the return ending at t+2.
market["equity_weight"] = market["signal"].shift(2).fillna(0.0)
market["cash_weight"] = 1.0 - market["equity_weight"]


# %% Calculate returns, implementation costs, and wealth

expense_factor = (1.0 - ANNUAL_ETF_EXPENSE) ** (
    market["calendar_days"] / 365.2425
)
market["equity_return_after_expense"] = (
    (1.0 + market["net_return"]) * expense_factor - 1.0
)

market["return_before_trading_cost"] = (
    market["equity_weight"] * market["equity_return_after_expense"]
    + market["cash_weight"] * market["cash_return"]
)
market["turnover"] = market["equity_weight"].diff().abs().fillna(0.0)
market["strategy_return"] = (
    market["return_before_trading_cost"]
    - TRANSACTION_COST * market["turnover"]
)

# Start the comparison once the 252-day signal and execution lag are available.
results = market.dropna(subset=["momentum_252"]).copy()
results["strategy_wealth"] = (1.0 + results["strategy_return"]).cumprod()
results["market_wealth"] = (
    1.0 + results["equity_return_after_expense"]
).cumprod()


# %% Plot the first strategy idea against ACWI buy-and-hold

plt.figure(figsize=(10, 5))
plt.plot(results["date"], results["strategy_wealth"], label="252-day TSMOM")
plt.plot(results["date"], results["market_wealth"], label="ACWI buy-and-hold")
plt.yscale("log")
plt.title("Initial 252-day time-series momentum prototype")
plt.ylabel("Growth of $1 (log scale)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()


# %% Calculate basic performance statistics

elapsed_years = (
    results["date"].iloc[-1] - results["date"].iloc[0]
).days / 365.2425
ending_wealth = results["strategy_wealth"].iloc[-1]
cagr = ending_wealth ** (1.0 / elapsed_years) - 1.0
volatility = results["strategy_return"].std(ddof=1) * (252**0.5)
excess_return = results["strategy_return"] - results["cash_return"]
sharpe = excess_return.mean() / excess_return.std(ddof=1) * (252**0.5)
drawdown = results["strategy_wealth"] / results["strategy_wealth"].cummax() - 1.0
max_drawdown = drawdown.min()

print(f"CAGR:         {cagr:.2%}")
print(f"Volatility:   {volatility:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Max drawdown: {max_drawdown:.2%}")
