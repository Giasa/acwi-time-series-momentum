"""MSCI ACWI absolute time-series momentum parameter study.

Research question
-----------------
Does time-series momentum work on MSCI ACWI, and how does performance vary
across momentum lookbacks and holding/rebalancing periods?

The script uses pre-2020 observations to select one specification, freezes that
choice, and then reports its performance from 2020 onward.  Run the file from
top to bottom, or execute the ``# %%`` sections individually in VS Code.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Keep Matplotlib's cache in a writable temporary directory.  This matters in
# restricted environments and has no effect on the research calculations.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tsmom-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# %% Research settings

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading-day lookbacks: approximately 3, 6, 9, 12, 15, and 24 months.
LOOKBACKS = [63, 126, 189, 252, 315, 504]
HOLDINGS = ["W", "M", "Q"]

POST_2020_START = pd.Timestamp("2020-01-01")
INITIAL_WEALTH = 10_000.0
TRANSACTION_COST_BPS = 5.0
ANNUAL_ETF_EXPENSE = 0.0032
TRADING_DAYS = 252


# %% Data loading and alignment

def load_data() -> pd.DataFrame:
    """Load and align the latest cached MSCI and FRED source files.

    Gross total return levels are retained for signal construction.  Net total
    return levels are retained for realized equity returns.  Treasury yields
    are aligned backward: each ACWI date receives only the latest rate that had
    already been published on or before that date.
    """

    gross_files = sorted(DATA_DIR.glob("msci_acwi_grtr_*.csv"))
    net_files = sorted(DATA_DIR.glob("msci_acwi_netr_*.csv"))
    fred_files = sorted(DATA_DIR.glob("fred_dgs3mo_*.csv"))
    if not gross_files or not net_files or not fred_files:
        raise FileNotFoundError(
            "Expected cached MSCI gross, MSCI net, and FRED CSV files in data/raw."
        )

    gross = pd.read_csv(gross_files[-1])
    net = pd.read_csv(net_files[-1])
    fred = pd.read_csv(fred_files[-1])

    for name, frame in (("gross MSCI", gross), ("net MSCI", net)):
        required = {"DATE", "LEVEL"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} data must contain DATE and LEVEL columns.")
        frame["DATE"] = pd.to_datetime(frame["DATE"], errors="raise")
        frame["LEVEL"] = pd.to_numeric(frame["LEVEL"], errors="raise")
        if frame["DATE"].duplicated().any():
            raise ValueError(f"{name} data contains duplicate dates.")
        if (frame["LEVEL"] <= 0).any():
            raise ValueError(f"{name} levels must be strictly positive.")

    gross = gross[["DATE", "LEVEL"]].rename(
        columns={"DATE": "date", "LEVEL": "gross_level"}
    )
    net = net[["DATE", "LEVEL"]].rename(
        columns={"DATE": "date", "LEVEL": "net_level"}
    )

    fred_date_column = (
        "observation_date" if "observation_date" in fred.columns else "DATE"
    )
    if fred_date_column not in fred.columns or "DGS3MO" not in fred.columns:
        raise ValueError("FRED data must contain a date column and DGS3MO.")
    fred = fred[[fred_date_column, "DGS3MO"]].rename(
        columns={fred_date_column: "date", "DGS3MO": "rf_annual_yield"}
    )
    fred["date"] = pd.to_datetime(fred["date"], errors="raise")
    fred["rf_annual_yield"] = pd.to_numeric(
        fred["rf_annual_yield"].replace(".", np.nan), errors="coerce"
    )
    fred = fred.dropna(subset=["rf_annual_yield"]).sort_values("date")
    if fred["date"].duplicated().any():
        raise ValueError("FRED data contains duplicate dates.")

    # Gross and net levels must refer to the same ACWI closing dates.
    market = gross.merge(net, on="date", how="inner", validate="one_to_one")
    market = market.sort_values("date")

    # merge_asof(direction="backward") is the key anti-look-ahead operation:
    # a Treasury observation dated after an ACWI close can never be used there.
    market = pd.merge_asof(
        market,
        fred,
        on="date",
        direction="backward",
        allow_exact_matches=True,
    ).dropna(subset=["rf_annual_yield"])

    market = market.set_index("date").sort_index()
    print(
        f"Loaded {len(market):,} aligned observations from "
        f"{market.index.min().date()} to {market.index.max().date()}."
    )
    print(f"Gross source: {gross_files[-1].name}")
    print(f"Net source:   {net_files[-1].name}")
    print(f"Cash source:  {fred_files[-1].name}")
    return market


def prepare_data(market: pd.DataFrame) -> pd.DataFrame:
    """Calculate net equity returns and calendar-day cash returns."""

    prepared = market.copy()
    prepared["calendar_days"] = (
        prepared.index.to_series().diff().dt.days.fillna(0)
    )
    prepared["equity_return"] = (
        prepared["net_level"].pct_change(fill_method=None).fillna(0.0)
    )

    # A return ending at close t uses the Treasury yield known at close t-1.
    prior_yield = prepared["rf_annual_yield"].shift(1).ffill()
    prior_yield.iloc[0] = prepared["rf_annual_yield"].iloc[0]
    prepared["cash_return"] = (
        (1.0 + prior_yield / 100.0)
        ** (prepared["calendar_days"] / 365.2425)
        - 1.0
    )
    return prepared


# %% Signal, position, and backtest logic

def build_signal(gross_level: pd.Series, lookback: int) -> pd.Series:
    """Return 1 when trailing gross total return is positive, otherwise 0."""

    momentum = gross_level.pct_change(lookback, fill_method=None)
    return (momentum > 0).astype(float).where(momentum.notna())


def build_position(
    gross_level: pd.Series, lookback: int, holding: str
) -> pd.Series:
    """Convert daily momentum observations into an executable 0/1 position.

    ``holding`` controls when the signal may be refreshed:

    - W: last ACWI observation of each calendar week
    - M: last ACWI observation of each calendar month
    - Q: last ACWI observation of each calendar quarter

    The two-row shift is deliberate.  A signal seen at close t is traded at
    close t+1, so the new position first earns the return ending at close t+2.
    """

    signal = build_signal(gross_level, lookback)
    period_frequency = {"W": "W-FRI", "M": "M", "Q": "Q"}
    if holding not in period_frequency:
        raise ValueError("holding must be one of: W, M, Q")

    periods = gross_level.index.to_period(period_frequency[holding])
    rebalance_dates = np.r_[periods[:-1] != periods[1:], True]
    signal_at_rebalance = signal.where(rebalance_dates).ffill().fillna(0.0)
    position = signal_at_rebalance.shift(2).fillna(0.0)
    return position.rename("equity_weight")


def run_backtest(
    market: pd.DataFrame,
    equity_position: pd.Series,
) -> pd.DataFrame:
    """Run a long/cash backtest including trading costs and ETF expenses."""

    backtest = market.copy()
    backtest["equity_weight"] = (
        equity_position.reindex(backtest.index).ffill().fillna(0.0).clip(0.0, 1.0)
    )
    backtest["cash_weight"] = 1.0 - backtest["equity_weight"]

    backtest["turnover"] = backtest["equity_weight"].diff().abs()
    backtest.iloc[0, backtest.columns.get_loc("turnover")] = abs(
        backtest["equity_weight"].iloc[0]
    )
    backtest["transaction_cost"] = (
        backtest["turnover"] * TRANSACTION_COST_BPS / 10_000.0
    )

    # The annual ETF expense is charged only while the portfolio holds ACWI.
    daily_expense_factor = (1.0 - ANNUAL_ETF_EXPENSE) ** (
        backtest["calendar_days"] / 365.2425
    )
    equity_return_after_expense = (
        (1.0 + backtest["equity_return"]) * daily_expense_factor - 1.0
    )
    return_before_trading_cost = (
        backtest["equity_weight"] * equity_return_after_expense
        + backtest["cash_weight"] * backtest["cash_return"]
    )
    backtest["strategy_return"] = (
        return_before_trading_cost - backtest["transaction_cost"]
    )
    backtest["wealth"] = INITIAL_WEALTH * (
        1.0 + backtest["strategy_return"]
    ).cumprod()
    backtest["drawdown"] = backtest["wealth"] / backtest["wealth"].cummax() - 1.0
    return backtest


def performance_metrics(backtest: pd.DataFrame) -> dict[str, float]:
    """Calculate the requested performance and trading statistics."""

    strategy_returns = backtest["strategy_return"]
    cash_returns = backtest["cash_return"]
    wealth = INITIAL_WEALTH * (1.0 + strategy_returns).cumprod()
    elapsed_years = max(
        (backtest.index[-1] - backtest.index[0]).days / 365.2425,
        1.0 / 365.2425,
    )
    excess_returns = strategy_returns - cash_returns
    excess_volatility = excess_returns.std(ddof=1)

    return {
        "cagr": float(
            (wealth.iloc[-1] / INITIAL_WEALTH) ** (1.0 / elapsed_years) - 1.0
        ),
        "volatility": float(strategy_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": float(
            excess_returns.mean() / excess_volatility * np.sqrt(TRADING_DAYS)
            if excess_volatility > 0
            else np.nan
        ),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "ending_wealth": float(wealth.iloc[-1]),
        "turnover": float(backtest["turnover"].sum()),
        "position_changes": int((backtest["turnover"] > 0).sum()),
    }


# %% Parameter grid and selection

def run_parameter_grid(
    market: pd.DataFrame,
    development_start: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[tuple[int, str], pd.DataFrame]]:
    """Evaluate every lookback/holding combination before 2020."""

    development = (market.index >= development_start) & (
        market.index < POST_2020_START
    )
    rows = []
    backtests = {}

    for lookback in LOOKBACKS:
        for holding in HOLDINGS:
            position = build_position(market["gross_level"], lookback, holding)
            backtest = run_backtest(market, position)
            metrics = performance_metrics(backtest.loc[development])
            rows.append({"lookback": lookback, "holding": holding, **metrics})
            backtests[(lookback, holding)] = backtest

    results = pd.DataFrame(rows).sort_values(
        ["sharpe", "turnover", "max_drawdown"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    results.insert(0, "development_rank", range(1, len(results) + 1))
    return results.reset_index(drop=True), backtests


# %% Charts

def plot_results(
    development_results: pd.DataFrame,
    selected_backtest: pd.DataFrame,
    buy_and_hold: pd.DataFrame,
    full_sample_start: pd.Timestamp,
) -> None:
    """Save the heatmap, wealth comparison, and drawdown comparison."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    heatmap = development_results.pivot(
        index="lookback", columns="holding", values="sharpe"
    ).reindex(index=LOOKBACKS, columns=HOLDINGS)
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    image = axis.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
    axis.set_xticks(range(len(HOLDINGS)), HOLDINGS)
    axis.set_yticks(range(len(LOOKBACKS)), LOOKBACKS)
    axis.set_xlabel("Holding / rebalance period")
    axis.set_ylabel("Momentum lookback (trading days)")
    axis.set_title("Development-period Sharpe ratio")
    for row_number in range(len(LOOKBACKS)):
        for column_number in range(len(HOLDINGS)):
            value = heatmap.iloc[row_number, column_number]
            text_color = "white" if value < heatmap.to_numpy().mean() else "black"
            axis.text(
                column_number,
                row_number,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
            )
    fig.colorbar(image, ax=axis, label="Sharpe ratio")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sharpe_heatmap.png", dpi=180)
    plt.close(fig)

    selected_returns = selected_backtest.loc[full_sample_start:, "strategy_return"]
    buy_hold_returns = buy_and_hold.loc[full_sample_start:, "strategy_return"]
    wealth = pd.DataFrame(
        {
            "Selected TSMOM": INITIAL_WEALTH * (1.0 + selected_returns).cumprod(),
            "ACWI buy-and-hold": INITIAL_WEALTH * (1.0 + buy_hold_returns).cumprod(),
        }
    )

    fig, axis = plt.subplots(figsize=(10, 5.5))
    wealth.plot(ax=axis, linewidth=1.5)
    axis.axvline(
        POST_2020_START,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Post-2020 evaluation begins",
    )
    axis.set_yscale("log")
    axis.set_title("Growth of $10,000: selected TSMOM vs ACWI")
    axis.set_ylabel("Portfolio value (log scale)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "wealth_curve.png", dpi=180)
    plt.close(fig)

    drawdowns = wealth / wealth.cummax() - 1.0
    fig, axis = plt.subplots(figsize=(10, 5))
    drawdowns.plot(ax=axis, linewidth=1.3)
    axis.axvline(
        POST_2020_START,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    axis.set_title("Drawdowns")
    axis.set_ylabel("Drawdown")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "drawdowns.png", dpi=180)
    plt.close(fig)


# %% Run the research

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    market_data = prepare_data(load_data())

    # Use one common evaluation start for every specification.  Wait until the
    # longest lookback has reached its first quarter-end decision and then add
    # the two-session execution lag.  All 18 strategies are operational by this
    # date, so longer lookbacks are not penalized by a longer warm-up period.
    longest_signal = build_signal(market_data["gross_level"], max(LOOKBACKS))
    quarters = market_data.index.to_period("Q")
    quarter_ends = np.r_[quarters[:-1] != quarters[1:], True]
    first_longest_quarter_end = np.flatnonzero(
        quarter_ends & longest_signal.notna().to_numpy()
    )[0]
    development_start = market_data.index[first_longest_quarter_end + 2]

    development_results, all_backtests = run_parameter_grid(
        market_data, development_start
    )
    selected_row = development_results.iloc[0]
    selected_lookback = int(selected_row["lookback"])
    selected_holding = str(selected_row["holding"])
    selected_backtest = all_backtests[(selected_lookback, selected_holding)]

    # Buy-and-hold uses the same net-return proxy, ETF expense, and transaction
    # cost model.  Its position is always 100% equity.
    buy_and_hold_position = pd.Series(1.0, index=market_data.index)
    buy_and_hold = run_backtest(market_data, buy_and_hold_position)

    period_masks = {
        "Development": (market_data.index >= development_start)
        & (market_data.index < POST_2020_START),
        "Post-2020 evaluation": market_data.index >= POST_2020_START,
        "Full sample": market_data.index >= development_start,
    }
    comparison_rows = []
    for period_name, period_mask in period_masks.items():
        for strategy_name, backtest in (
            ("Selected TSMOM", selected_backtest),
            ("ACWI buy-and-hold", buy_and_hold),
        ):
            comparison_rows.append(
                {
                    "period": period_name,
                    "strategy": strategy_name,
                    **performance_metrics(backtest.loc[period_mask]),
                }
            )
    period_comparison = pd.DataFrame(comparison_rows)

    selected_specification = pd.DataFrame(
        [
            {
                "lookback_trading_days": selected_lookback,
                "holding_period": selected_holding,
                "selection_period_start": development_start.date(),
                "selection_period_end": (POST_2020_START - pd.Timedelta(days=1)).date(),
                "selection_rule": "Highest development-period Sharpe ratio",
            }
        ]
    )

    development_results.to_csv(
        RESULTS_DIR / "development_results.csv", index=False
    )
    period_comparison.to_csv(
        RESULTS_DIR / "period_comparison.csv", index=False
    )
    selected_specification.to_csv(
        RESULTS_DIR / "selected_strategy.csv", index=False
    )
    plot_results(
        development_results,
        selected_backtest,
        buy_and_hold,
        development_start,
    )

    display_columns = [
        "development_rank",
        "lookback",
        "holding",
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
        "turnover",
        "position_changes",
    ]
    print("\nDevelopment-period parameter study")
    print(development_results[display_columns].to_string(index=False))
    print("\nSelected specification")
    print(selected_specification.to_string(index=False))
    print("\nPeriod comparison")
    print(period_comparison.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_DIR}")
