"""
Tests for Experiment 2: Static Portfolio Weight Computation.

Validates:
- Equal Weight: weights = 1/n, sum = 1
- Minimum Variance: feasibility, optimality bounds
- Maximum Diversification: DR > 1 for uncorrelated assets
- Equal Risk Contribution: equal TRC across assets
- Strategy return series properties
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio_static import (
    _erc_objective,
    _neg_diversification_ratio,
    _portfolio_variance,
    compute_all_static_portfolios,
    compute_equal_weight,
    compute_erc,
    compute_max_diversification,
    compute_min_variance,
    compute_strategy_returns,
    validate_weights,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_returns():
    """Simple returns DataFrame: 5 assets, 500 days."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=500, freq="B")
    n_assets = 5
    corr = np.array([
        [1.0, 0.3, 0.1, 0.2, 0.4],
        [0.3, 1.0, 0.2, 0.1, 0.3],
        [0.1, 0.2, 1.0, 0.5, 0.2],
        [0.2, 0.1, 0.5, 1.0, 0.1],
        [0.4, 0.3, 0.2, 0.1, 1.0],
    ])
    vols = np.array([0.01, 0.015, 0.008, 0.012, 0.02])
    cov = np.diag(vols) @ corr @ np.diag(vols)
    L = np.linalg.cholesky(cov)
    raw = np.random.randn(500, n_assets) @ L.T
    returns = pd.DataFrame(raw, index=dates, columns=[f"A{i}" for i in range(n_assets)])
    return returns


@pytest.fixture
def high_vol_returns():
    """Returns with one very high-volatility asset (simulating BTC)."""
    np.random.seed(7)
    dates = pd.date_range("2020-01-02", periods=300, freq="B")
    returns = pd.DataFrame({
        "EQ1": np.random.randn(300) * 0.01,
        "EQ2": np.random.randn(300) * 0.012,
        "HIGH": np.random.randn(300) * 0.05,  # 5x higher vol
    }, index=dates)
    return returns


# ── Equal Weight tests ────────────────────────────────────────────────────────

def test_equal_weight_sum(simple_returns):
    """EW weights must sum to 1."""
    n = simple_returns.shape[1]
    w = compute_equal_weight(n)
    assert abs(w.sum() - 1.0) < 1e-12


def test_equal_weight_values(simple_returns):
    """Each EW weight should be exactly 1/n."""
    n = simple_returns.shape[1]
    w = compute_equal_weight(n)
    expected = 1.0 / n
    np.testing.assert_allclose(w, expected, rtol=1e-12)


def test_equal_weight_shape(simple_returns):
    """EW weight vector should have length n."""
    n = simple_returns.shape[1]
    w = compute_equal_weight(n)
    assert len(w) == n


# ── Minimum Variance tests ────────────────────────────────────────────────────

def test_minvar_weights_sum(simple_returns):
    """MinVar weights must sum to 1."""
    w, success = compute_min_variance(simple_returns)
    assert abs(w.sum() - 1.0) < 1e-6


def test_minvar_weights_non_negative(simple_returns):
    """MinVar weights must be non-negative (long-only)."""
    w, _ = compute_min_variance(simple_returns)
    assert np.all(w >= -1e-8)


def test_minvar_convergence(simple_returns):
    """MinVar optimizer should converge."""
    _, success = compute_min_variance(simple_returns)
    assert success


def test_minvar_lower_variance_than_ew(simple_returns):
    """MinVar portfolio variance should be <= EW portfolio variance."""
    n = simple_returns.shape[1]
    w_ew = compute_equal_weight(n)
    w_mv, _ = compute_min_variance(simple_returns)

    Sigma = simple_returns.cov().values
    var_mv = w_mv @ Sigma @ w_mv
    var_ew = w_ew @ Sigma @ w_ew

    assert var_mv <= var_ew + 1e-8


def test_minvar_high_vol_asset_low_weight(high_vol_returns):
    """MinVar should assign near-zero weight to the high-vol asset."""
    w, success = compute_min_variance(high_vol_returns)
    high_vol_idx = 2  # "HIGH" asset
    assert w[high_vol_idx] < 0.2, f"High-vol asset weight: {w[high_vol_idx]:.4f}"


# ── Maximum Diversification tests ─────────────────────────────────────────────

def test_maxdiv_weights_sum(simple_returns):
    """MaxDiv weights must sum to 1."""
    w, _ = compute_max_diversification(simple_returns)
    assert abs(w.sum() - 1.0) < 1e-6


def test_maxdiv_weights_non_negative(simple_returns):
    """MaxDiv weights must be non-negative."""
    w, _ = compute_max_diversification(simple_returns)
    assert np.all(w >= -1e-8)


def test_maxdiv_dr_greater_than_one(simple_returns):
    """Diversification ratio should be > 1 for a non-perfectly-correlated portfolio."""
    w, _ = compute_max_diversification(simple_returns)
    Sigma = simple_returns.cov().values + 1e-8 * np.eye(simple_returns.shape[1])
    sigma_vec = np.sqrt(np.diag(Sigma))
    dr = np.dot(w, sigma_vec) / np.sqrt(w @ Sigma @ w)
    assert dr > 1.0


def test_maxdiv_high_vol_asset_low_weight(high_vol_returns):
    """MaxDiv should assign low weight to the asset with disproportionate vol."""
    w, _ = compute_max_diversification(high_vol_returns)
    # High-vol asset should not dominate
    high_vol_idx = 2
    assert w[high_vol_idx] < 0.5


# ── Equal Risk Contribution tests ─────────────────────────────────────────────

def test_erc_weights_sum(simple_returns):
    """ERC weights must sum to 1."""
    w, _ = compute_erc(simple_returns)
    assert abs(w.sum() - 1.0) < 1e-6


def test_erc_weights_non_negative(simple_returns):
    """ERC weights must be non-negative."""
    w, _ = compute_erc(simple_returns)
    assert np.all(w >= -1e-8)


def test_erc_risk_contributions_equal(simple_returns):
    """TRC_i = w_i * (Sigma @ w)_i should be approximately equal for all i."""
    w, _ = compute_erc(simple_returns)
    Sigma = simple_returns.cov().values + 1e-8 * np.eye(simple_returns.shape[1])
    trc = w * (Sigma @ w)
    trc_range = trc.max() - trc.min()
    assert trc_range < 1e-5, f"TRC range too large: {trc_range:.2e}"


def test_erc_high_vol_lower_weight(high_vol_returns):
    """ERC should assign lower weight to higher-volatility asset."""
    w, _ = compute_erc(high_vol_returns)
    high_vol_idx = 2
    eq_idx = 0
    assert w[high_vol_idx] < w[eq_idx], (
        f"High-vol weight ({w[high_vol_idx]:.4f}) should be < low-vol weight ({w[eq_idx]:.4f})"
    )


# ── Objective function tests ──────────────────────────────────────────────────

def test_portfolio_variance_positive(simple_returns):
    """Portfolio variance should be positive for any non-zero weight vector."""
    n = simple_returns.shape[1]
    w = np.ones(n) / n
    Sigma = simple_returns.cov().values
    var = _portfolio_variance(w, Sigma)
    assert var > 0


def test_erc_objective_zero_at_solution(simple_returns):
    """ERC objective should be ~0 at the true ERC solution."""
    w, _ = compute_erc(simple_returns)
    Sigma = simple_returns.cov().values + 1e-8 * np.eye(simple_returns.shape[1])
    obj_val = _erc_objective(w, Sigma)
    assert obj_val < 1e-10, f"ERC objective at solution: {obj_val:.2e}"


def test_neg_dr_negative(simple_returns):
    """Negative DR should be <= 0 (actual DR >= 0)."""
    n = simple_returns.shape[1]
    w = np.ones(n) / n
    Sigma = simple_returns.cov().values + 1e-8 * np.eye(n)
    sigma_vec = np.sqrt(np.diag(Sigma))
    neg_dr = _neg_diversification_ratio(w, Sigma, sigma_vec)
    assert neg_dr <= 0


# ── Strategy return series tests ──────────────────────────────────────────────

def test_strategy_returns_shape(simple_returns):
    """Strategy returns DataFrame should have same number of rows as input returns."""
    n = simple_returns.shape[1]
    weights = {
        "EW": np.ones(n) / n,
        "Test": np.array([0.5, 0.2, 0.1, 0.1, 0.1]),
    }
    strat_rets = compute_strategy_returns(simple_returns, weights)
    assert len(strat_rets) == len(simple_returns)
    assert set(strat_rets.columns) == {"EW", "Test"}


def test_strategy_returns_ew_matches_mean(simple_returns):
    """EW strategy return should equal the row mean of asset returns."""
    n = simple_returns.shape[1]
    weights = {"EW": np.ones(n) / n}
    strat_rets = compute_strategy_returns(simple_returns, weights)
    expected = simple_returns.mean(axis=1)
    np.testing.assert_allclose(strat_rets["EW"].values, expected.values, rtol=1e-10)


# ── Full pipeline tests ───────────────────────────────────────────────────────

def test_all_static_portfolios_returns_4_strategies(simple_returns):
    """compute_all_static_portfolios should return exactly 4 strategy weights."""
    weights, _ = compute_all_static_portfolios(simple_returns)
    assert set(weights.keys()) == {"EW", "MinVar", "MaxDiv", "ERC"}


def test_all_static_portfolios_all_sums_to_one(simple_returns):
    """All four weight vectors should sum to 1."""
    weights, _ = compute_all_static_portfolios(simple_returns)
    for name, w in weights.items():
        assert abs(w.sum() - 1.0) < 1e-6, f"{name} weights sum = {w.sum():.8f}"


def test_all_static_portfolios_returns_df_shape(simple_returns):
    """Strategy returns DataFrame should have 4 columns and same length as input."""
    _, strat_rets = compute_all_static_portfolios(simple_returns)
    assert strat_rets.shape == (len(simple_returns), 4)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_equal_weight_single_asset():
    """Equal weight for 1 asset should be 1.0."""
    w = compute_equal_weight(1)
    assert abs(w[0] - 1.0) < 1e-12


def test_minvar_two_assets():
    """MinVar should work with just 2 assets."""
    dates = pd.date_range("2020-01-02", periods=200, freq="B")
    rets = pd.DataFrame({
        "A": np.random.randn(200) * 0.01,
        "B": np.random.randn(200) * 0.02,
    }, index=dates)
    w, success = compute_min_variance(rets)
    assert success
    assert abs(w.sum() - 1.0) < 1e-6


def test_validate_weights_passes(simple_returns):
    """validate_weights should not raise for valid weights."""
    n = simple_returns.shape[1]
    w = np.ones(n) / n
    # Should not raise
    validate_weights(w, "EW", simple_returns)
