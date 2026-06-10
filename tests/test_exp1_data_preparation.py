"""
Tests for Experiment 1: Data Preparation and Market State Classification.

Validates:
- Data loading and alignment
- Return computation
- Rolling volatility
- K-means state assignment
- State distribution properties
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    align_prices,
    compute_equal_weight_portfolio,
    compute_returns,
    compute_rolling_volatility,
)
from src.market_states import (
    build_state_series,
    fit_kmeans_states,
    validate_state_distribution,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_prices():
    """Generate synthetic prices for testing (20 assets, 500 days)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=500, freq="B")
    n_assets = 5
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(500, n_assets) * 0.01, axis=0) * 100,
        index=dates,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    return prices


@pytest.fixture
def synthetic_returns(synthetic_prices):
    """Simple returns from synthetic prices."""
    return compute_returns(synthetic_prices)


@pytest.fixture
def synthetic_vol(synthetic_returns):
    """Rolling volatility from synthetic returns."""
    eq = compute_equal_weight_portfolio(synthetic_returns)
    return compute_rolling_volatility(eq, window=22)


# ── Data alignment tests ──────────────────────────────────────────────────────

def test_align_prices_no_nan(synthetic_prices):
    """Aligned prices must have zero NaN values."""
    # Inject some NaN
    prices_with_nan = synthetic_prices.copy()
    prices_with_nan.iloc[5, 2] = np.nan

    aligned = align_prices(prices_with_nan)
    assert aligned.isna().sum().sum() == 0


def test_align_prices_row_count_decreases(synthetic_prices):
    """Alignment drops rows with NaN, so row count <= original."""
    prices_with_nan = synthetic_prices.copy()
    prices_with_nan.iloc[0:3, 0] = np.nan

    aligned = align_prices(prices_with_nan)
    assert len(aligned) <= len(synthetic_prices)
    assert len(aligned) >= len(synthetic_prices) - 3


def test_align_prices_perfect_data(synthetic_prices):
    """Alignment should not drop any rows when no NaN exists."""
    aligned = align_prices(synthetic_prices)
    assert len(aligned) == len(synthetic_prices)


# ── Return computation tests ──────────────────────────────────────────────────

def test_returns_shape(synthetic_prices, synthetic_returns):
    """Returns should have one fewer row than prices (due to pct_change)."""
    assert len(synthetic_returns) == len(synthetic_prices) - 1


def test_returns_no_nan(synthetic_returns):
    """After dropping first NaN row, no NaN should remain."""
    assert synthetic_returns.isna().sum().sum() == 0


def test_returns_range(synthetic_returns):
    """Daily returns should be in a reasonable range (-50% to +50%)."""
    assert synthetic_returns.values.max() < 0.5
    assert synthetic_returns.values.min() > -0.5


def test_equal_weight_mean(synthetic_returns):
    """Equal-weighted portfolio return should equal the row mean."""
    eq_ret = compute_equal_weight_portfolio(synthetic_returns)
    expected = synthetic_returns.mean(axis=1)
    np.testing.assert_allclose(eq_ret.values, expected.values, rtol=1e-10)


def test_equal_weight_series_name(synthetic_returns):
    """Equal-weighted series should have a proper name."""
    eq_ret = compute_equal_weight_portfolio(synthetic_returns)
    assert eq_ret.name is not None


# ── Rolling volatility tests ──────────────────────────────────────────────────

def test_rolling_vol_warmup_nan(synthetic_returns):
    """First 21 rows of rolling vol should be NaN (window=22)."""
    eq = compute_equal_weight_portfolio(synthetic_returns)
    vol = compute_rolling_volatility(eq, window=22)
    assert vol.iloc[:21].isna().all()


def test_rolling_vol_valid_from_row22(synthetic_returns):
    """Row 22 onward should have valid (non-NaN) volatility values."""
    eq = compute_equal_weight_portfolio(synthetic_returns)
    vol = compute_rolling_volatility(eq, window=22)
    assert not vol.iloc[21:].isna().any()


def test_rolling_vol_positive(synthetic_returns):
    """All valid rolling volatility values should be strictly positive."""
    eq = compute_equal_weight_portfolio(synthetic_returns)
    vol = compute_rolling_volatility(eq, window=22)
    valid = vol.dropna()
    assert (valid > 0).all()


def test_rolling_vol_matches_manual(synthetic_returns):
    """Rolling vol should match pandas manual rolling std calculation."""
    eq = compute_equal_weight_portfolio(synthetic_returns)
    vol = compute_rolling_volatility(eq, window=22)
    manual = eq.rolling(22, min_periods=22).std(ddof=1)
    pd.testing.assert_series_equal(vol, manual, check_names=False)


def test_rolling_vol_window_size(synthetic_returns):
    """The number of valid vol values should be len(returns) - 21."""
    eq = compute_equal_weight_portfolio(synthetic_returns)
    vol = compute_rolling_volatility(eq, window=22)
    n_valid = vol.dropna().shape[0]
    expected = len(synthetic_returns) - 21
    assert n_valid == expected


# ── K-means clustering tests ──────────────────────────────────────────────────

def test_kmeans_state_count(synthetic_vol):
    """K-means should assign exactly k=5 states (test with k=5)."""
    states, centroids, valid_vol, km = fit_kmeans_states(
        synthetic_vol, n_clusters=5, n_init=10, max_iter=200, random_state=0
    )
    unique_states = np.unique(states)
    assert len(unique_states) == 5


def test_kmeans_state_range(synthetic_vol):
    """State labels should be integers in {1, ..., n_clusters}."""
    k = 5
    states, _, _, _ = fit_kmeans_states(synthetic_vol, n_clusters=k, n_init=10)
    assert states.min() >= 1
    assert states.max() <= k


def test_kmeans_centroids_ordered(synthetic_vol):
    """Centroids should be in ascending order."""
    _, centroids, _, _ = fit_kmeans_states(synthetic_vol, n_clusters=5, n_init=10)
    assert np.all(np.diff(centroids) >= 0), "Centroids not in ascending order"


def test_kmeans_state_length(synthetic_vol):
    """Number of states should match valid volatility observations."""
    states, _, valid_vol, _ = fit_kmeans_states(synthetic_vol, n_clusters=5, n_init=10)
    assert len(states) == len(valid_vol)
    assert len(states) == synthetic_vol.dropna().shape[0]


def test_kmeans_state_1_lowest_vol(synthetic_vol):
    """State 1 should correspond to the lowest-volatility centroid."""
    states, centroids, valid_vol, _ = fit_kmeans_states(
        synthetic_vol, n_clusters=5, n_init=20, random_state=42
    )
    # Centroid for state 1 should be less than centroid for state 5
    assert centroids[0] < centroids[-1]


def test_build_state_series_length(synthetic_vol):
    """State series length should match valid vol observations."""
    states, _, _, _ = fit_kmeans_states(synthetic_vol, n_clusters=5, n_init=10)
    state_series = build_state_series(synthetic_vol, states)
    assert len(state_series) == len(synthetic_vol.dropna())


def test_build_state_series_index(synthetic_vol):
    """State series index should align with valid vol dates."""
    states, _, _, _ = fit_kmeans_states(synthetic_vol, n_clusters=5, n_init=10)
    state_series = build_state_series(synthetic_vol, states)
    expected_dates = synthetic_vol.dropna().index
    pd.testing.assert_index_equal(state_series.index, expected_dates)


def test_build_state_series_dtype(synthetic_vol):
    """State series values should be integers."""
    states, _, _, _ = fit_kmeans_states(synthetic_vol, n_clusters=5, n_init=10)
    state_series = build_state_series(synthetic_vol, states)
    assert state_series.dtype in [np.int32, np.int64, int]


# ── State distribution validation ────────────────────────────────────────────

def test_validate_state_distribution_returns_dict(synthetic_vol):
    """validate_state_distribution should return a dict."""
    states, _, _, _ = fit_kmeans_states(
        synthetic_vol, n_clusters=5, n_init=10, random_state=42
    )
    state_series = build_state_series(synthetic_vol, states)
    result = validate_state_distribution(state_series, n_clusters=5)
    assert isinstance(result, dict)


def test_state_distribution_sums_to_total(synthetic_vol):
    """Sum of all state counts should equal total length of state series."""
    states, _, _, _ = fit_kmeans_states(
        synthetic_vol, n_clusters=5, n_init=10, random_state=42
    )
    state_series = build_state_series(synthetic_vol, states)
    counts = state_series.value_counts()
    assert counts.sum() == len(state_series)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_single_asset_returns():
    """Test with a single asset (n=1)."""
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(100, 1) * 0.01) * 100,
        index=dates,
        columns=["A0"],
    )
    rets = compute_returns(prices)
    eq = compute_equal_weight_portfolio(rets)
    assert len(eq) == 99


def test_all_zero_returns_vol():
    """Rolling vol of all-zero returns should be zero."""
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    zero_returns = pd.Series(np.zeros(100), index=dates)
    vol = compute_rolling_volatility(zero_returns, window=22)
    valid = vol.dropna()
    np.testing.assert_allclose(valid.values, 0.0, atol=1e-15)


def test_extreme_returns():
    """Test with returns that include extreme values (+/-50%)."""
    np.random.seed(99)
    dates = pd.date_range("2020-01-02", periods=200, freq="B")
    extreme = pd.Series(
        np.random.choice([-0.5, 0.0, 0.5], size=200),
        index=dates,
    )
    vol = compute_rolling_volatility(extreme, window=22)
    valid = vol.dropna()
    assert (valid >= 0).all()
