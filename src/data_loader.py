"""
Data loading and preprocessing module.

Downloads, aligns, and computes returns for both asset sets using yfinance.
"""
import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Asset universe definitions ──────────────────────────────────────────────
ASSET_SET_1_TICKERS = ["AAPL", "LLY", "JPM", "AMZN", "GOOGL", "UPS", "PG", "XOM", "NEE", "AMT", "LIN"]
ASSET_SET_2_TICKERS = ["QQQ", "SPY", "BTC-USD", "GLD", "TLT"]

ASSET_SET_1_START = "2005-06-20"
ASSET_SET_1_END = "2024-06-20"

ASSET_SET_2_START = "2015-01-06"
ASSET_SET_2_END = "2024-06-20"

# Download buffer: start 30 calendar days before analysis start
ASSET_SET_1_DOWNLOAD_START = "2005-05-01"
ASSET_SET_2_DOWNLOAD_START = "2014-11-20"

DATA_DIR = Path(__file__).parent.parent / "data"


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download daily adjusted closing prices via yfinance.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start : str
        Start date (YYYY-MM-DD).  Include buffer before analysis window.
    end : str
        End date (YYYY-MM-DD), exclusive in yfinance convention.
    auto_adjust : bool
        Use auto-adjusted prices (accounts for splits and dividends).

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices with dates as index and
        tickers as columns.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    # yfinance ≥ 0.2 returns MultiIndex columns when multiple tickers are given
    if isinstance(raw.columns, pd.MultiIndex):
        # Try 'Close' first (auto_adjust=True uses 'Close' in newer versions)
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"]
        else:
            raise ValueError(f"Neither 'Close' nor 'Adj Close' found. Columns: {raw.columns.tolist()}")
    else:
        # Single ticker
        if "Close" in raw.columns:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        elif "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        else:
            raise ValueError(f"Expected 'Close' or 'Adj Close'. Got: {raw.columns.tolist()}")

    # Ensure correct column ordering
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available]
    return prices


def align_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Align price data by dropping any row with a NaN in any column (inner join).

    Parameters
    ----------
    prices : pd.DataFrame
        Raw price DataFrame (possibly containing NaN).

    Returns
    -------
    pd.DataFrame
        Fully aligned prices with no NaN values.
    """
    aligned = prices.dropna(axis=0, how="any")
    assert aligned.isna().sum().sum() == 0, "NaN values remain after alignment"
    return aligned


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns: r_{i,t} = P_{i,t}/P_{i,t-1} - 1.

    Parameters
    ----------
    prices : pd.DataFrame
        Aligned price DataFrame.

    Returns
    -------
    pd.DataFrame
        Simple daily returns with first row dropped (NaN due to pct_change).
    """
    returns = prices.pct_change()
    returns = returns.dropna(how="all")
    return returns


def compute_equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Compute equal-weighted portfolio return: r_eq,t = mean(r_{i,t}).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns DataFrame.

    Returns
    -------
    pd.Series
        Equal-weighted portfolio return series.
    """
    n = returns.shape[1]
    eq_ret = returns.mean(axis=1)
    eq_ret.name = "EqualWeight_Portfolio"
    return eq_ret


def compute_rolling_volatility(eq_returns: pd.Series, window: int = 22) -> pd.Series:
    """
    Compute 22-day rolling sample standard deviation of portfolio returns.

    This is the sole feature used for K-means market state clustering.

    Parameters
    ----------
    eq_returns : pd.Series
        Equal-weighted portfolio daily return series.
    window : int
        Rolling window size in trading days (default 22).

    Returns
    -------
    pd.Series
        Rolling volatility series sigma_{22,t}; first (window-1) values are NaN.
    """
    vol = eq_returns.rolling(window=window, min_periods=window).std(ddof=1)
    vol.name = "Rolling_Vol_22d"
    return vol


def load_asset_set_1(cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load, align, and preprocess Asset Set 1.

    Returns
    -------
    prices : pd.DataFrame
    returns : pd.DataFrame
    eq_returns : pd.Series
    rolling_vol : pd.Series
    """
    cache_path = DATA_DIR / "asset_set1_prices.parquet"

    if cache and cache_path.exists():
        prices_raw = pd.read_parquet(cache_path)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        prices_raw = download_prices(
            ASSET_SET_1_TICKERS,
            start=ASSET_SET_1_DOWNLOAD_START,
            end="2024-06-21",  # yfinance end is exclusive
        )
        prices_raw.to_parquet(cache_path)

    prices = align_prices(prices_raw)
    # Restrict to analysis window AFTER alignment (but keep buffer for rolling vol)
    analysis_start = pd.Timestamp(ASSET_SET_1_START)
    # Keep from download start to compute rolling vol from analysis start
    prices = prices[prices.index >= pd.Timestamp(ASSET_SET_1_DOWNLOAD_START)]
    prices = prices[prices.index <= pd.Timestamp(ASSET_SET_1_END)]

    returns = compute_returns(prices)
    eq_returns = compute_equal_weight_portfolio(returns)
    rolling_vol = compute_rolling_volatility(eq_returns, window=22)

    return prices, returns, eq_returns, rolling_vol


def load_asset_set_2(cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load, align, and preprocess Asset Set 2.

    Returns
    -------
    prices : pd.DataFrame
    returns : pd.DataFrame
    eq_returns : pd.Series
    rolling_vol : pd.Series
    """
    cache_path = DATA_DIR / "asset_set2_prices.parquet"

    if cache and cache_path.exists():
        prices_raw = pd.read_parquet(cache_path)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        prices_raw = download_prices(
            ASSET_SET_2_TICKERS,
            start=ASSET_SET_2_DOWNLOAD_START,
            end="2024-06-21",
        )
        prices_raw.to_parquet(cache_path)

    prices = align_prices(prices_raw)
    prices = prices[prices.index >= pd.Timestamp(ASSET_SET_2_DOWNLOAD_START)]
    prices = prices[prices.index <= pd.Timestamp(ASSET_SET_2_END)]

    returns = compute_returns(prices)
    eq_returns = compute_equal_weight_portfolio(returns)
    rolling_vol = compute_rolling_volatility(eq_returns, window=22)

    return prices, returns, eq_returns, rolling_vol


def get_analysis_period_data(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    eq_returns: pd.Series,
    rolling_vol: pd.Series,
    analysis_start: str,
    analysis_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Trim all data series to the official analysis window.

    The rolling vol series starts with NaN for the first 21 rows; after trimming
    to analysis_start, the first valid vol value is at the 22nd row of returns.

    Parameters
    ----------
    prices, returns, eq_returns, rolling_vol : DataFrames / Series
    analysis_start, analysis_end : str
        Official analysis window dates.

    Returns
    -------
    Tuple of (prices, returns, eq_returns, rolling_vol) trimmed to analysis window.
    """
    start = pd.Timestamp(analysis_start)
    end = pd.Timestamp(analysis_end)

    p = prices[(prices.index >= start) & (prices.index <= end)]
    r = returns[(returns.index >= start) & (returns.index <= end)]
    eq = eq_returns[(eq_returns.index >= start) & (eq_returns.index <= end)]
    vol = rolling_vol[(rolling_vol.index >= start) & (rolling_vol.index <= end)]

    return p, r, eq, vol
