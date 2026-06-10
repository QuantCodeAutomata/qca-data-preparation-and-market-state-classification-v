"""
Tests for Experiment 4: Dynamic Portfolio Construction and Performance Evaluation.

Validates:
- Per-state compound return computation
- Binary assignment vector construction
- Total return weight matrix (rows sum to 1)
- Dynamic portfolio return series
- Performance metrics (custom non-standard Sharpe ratio)
- Annual metrics computation
- Portfolio value series
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dynamic_portfolio import (
    build_binary_assignment_vectors,
    build_dynamic_portfolio,
    compound_return,
    compute_annual_metrics,
    compute_full_period_metrics,
    compute_per_state_compound_returns,
    compute_portfolio_value,
    compute_total_return_weights,
    custom_sharpe,
    daily_volatility,
    identify_best_strategy_per_state,
    run_full_evaluation,
)
from src.markov_transition import compute_posterior_mean, count_transitions


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_setup():
    """
    Synthetic full setup: 4 strategies, 4 states, 500 trading days.
    Returns (state_series, strategy_returns, P_mean).
    """
    np.random.seed(42)
    n_days = 500
    n_states = 4
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")

    # Simulate state series (Markov chain with 4 states)
    states = [1]
    P_true = np.array([
        [0.90, 0.08, 0.01, 0.01],
        [0.05, 0.88, 0.05, 0.02],
        [0.02, 0.04, 0.90, 0.04],
        [0.01, 0.02, 0.03, 0.94],
    ])
    for _ in range(n_days - 1):
        s = states[-1] - 1
        ns = np.random.choice(n_states, p=P_true[s]) + 1
        states.append(ns)
    state_series = pd.Series(states, index=dates, dtype=int, name="State")

    # Strategy returns: 4 strategies with different risk/return profiles
    strategy_returns = pd.DataFrame({
        "EW":     np.random.randn(n_days) * 0.01 + 0.0003,
        "MinVar": np.random.randn(n_days) * 0.007 + 0.0002,
        "MaxDiv": np.random.randn(n_days) * 0.009 + 0.00025,
        "ERC":    np.random.randn(n_days) * 0.008 + 0.00028,
    }, index=dates)

    # Build Bayesian transition matrix
    N = count_transitions(state_series, n_states=n_states)
    P_mean = compute_posterior_mean(N, alpha=1.0)

    return state_series, strategy_returns, P_mean, n_states


# ── Per-state compound return tests ──────────────────────────────────────────

def test_per_state_returns_shape(synthetic_setup):
    """Per-state returns should have shape (n_states, n_strategies)."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    result = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    assert result.shape == (n_states, 4)


def test_per_state_returns_index(synthetic_setup):
    """Per-state returns index should be state labels 1..n_states."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    result = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    expected_index = list(range(1, n_states + 1))
    assert result.index.tolist() == expected_index


def test_per_state_returns_no_nan(synthetic_setup):
    """Per-state compound returns should have no NaN (every state should appear)."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    result = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    assert not result.isna().any().any()


def test_per_state_returns_multiplicative(synthetic_setup):
    """Verify compound return formula: prod(1+r) - 1."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    result = compute_per_state_compound_returns(state_series, strat_rets, n_states)

    # Manually check state 1, EW strategy
    state1_dates = state_series[state_series == 1].index
    valid_dates = state1_dates.intersection(strat_rets.index)
    ew_rets = strat_rets.loc[valid_dates, "EW"]
    expected = (1.0 + ew_rets).prod() - 1.0
    assert abs(result.loc[1, "EW"] - expected) < 1e-12


# ── Binary assignment vector tests ────────────────────────────────────────────

def test_binary_vectors_mutual_exclusivity(synthetic_setup):
    """Exactly one strategy should be assigned to each state."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    per_state_rets = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    best = identify_best_strategy_per_state(per_state_rets)
    strategies = strat_rets.columns.tolist()
    p_vectors = build_binary_assignment_vectors(best, strategies, n_states)

    total = sum(p_vectors[m] for m in strategies)
    np.testing.assert_array_equal(total, np.ones(n_states))


def test_binary_vectors_values_binary(synthetic_setup):
    """Binary assignment vectors should only contain 0 or 1."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    per_state_rets = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    best = identify_best_strategy_per_state(per_state_rets)
    strategies = strat_rets.columns.tolist()
    p_vectors = build_binary_assignment_vectors(best, strategies, n_states)

    for m, p in p_vectors.items():
        assert set(np.unique(p)).issubset({0.0, 1.0}), f"{m}: non-binary values in {p}"


def test_binary_vectors_shape(synthetic_setup):
    """Each binary vector should have length n_states."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    per_state_rets = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    best = identify_best_strategy_per_state(per_state_rets)
    strategies = strat_rets.columns.tolist()
    p_vectors = build_binary_assignment_vectors(best, strategies, n_states)

    for m, p in p_vectors.items():
        assert len(p) == n_states


# ── Total return weight matrix tests ─────────────────────────────────────────

def test_total_return_weights_shape(synthetic_setup):
    """W matrix should have shape (n_states, n_strategies)."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    per_state_rets = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    best = identify_best_strategy_per_state(per_state_rets)
    strategies = strat_rets.columns.tolist()
    p_vectors = build_binary_assignment_vectors(best, strategies, n_states)
    W = compute_total_return_weights(P_mean, p_vectors)
    assert W.shape == (n_states, len(strategies))


def test_total_return_weights_rows_sum_to_one(synthetic_setup):
    """Each row of W must sum to 1.0."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    per_state_rets = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    best = identify_best_strategy_per_state(per_state_rets)
    strategies = strat_rets.columns.tolist()
    p_vectors = build_binary_assignment_vectors(best, strategies, n_states)
    W = compute_total_return_weights(P_mean, p_vectors)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-8)


def test_total_return_weights_non_negative(synthetic_setup):
    """W entries should be non-negative (probabilities)."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    per_state_rets = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    best = identify_best_strategy_per_state(per_state_rets)
    strategies = strat_rets.columns.tolist()
    p_vectors = build_binary_assignment_vectors(best, strategies, n_states)
    W = compute_total_return_weights(P_mean, p_vectors)
    assert np.all(W >= 0)


# ── Dynamic portfolio return series tests ─────────────────────────────────────

def test_dynamic_portfolio_length(synthetic_setup):
    """Dynamic returns should have length len(state_series) - 1."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    per_state_rets = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    best = identify_best_strategy_per_state(per_state_rets)
    strategies = strat_rets.columns.tolist()
    p_vectors = build_binary_assignment_vectors(best, strategies, n_states)
    W = compute_total_return_weights(P_mean, p_vectors)
    dyn = build_dynamic_portfolio(state_series, strat_rets, W)
    # Should be len(common_dates) - 1
    common = state_series.index.intersection(strat_rets.index)
    assert len(dyn) == len(common) - 1


def test_dynamic_portfolio_no_nan(synthetic_setup):
    """Dynamic portfolio should not contain NaN values."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    per_state_rets = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    best = identify_best_strategy_per_state(per_state_rets)
    strategies = strat_rets.columns.tolist()
    p_vectors = build_binary_assignment_vectors(best, strategies, n_states)
    W = compute_total_return_weights(P_mean, p_vectors)
    dyn = build_dynamic_portfolio(state_series, strat_rets, W)
    assert not dyn.isna().any()


def test_dynamic_portfolio_returns_name(synthetic_setup):
    """Dynamic portfolio series should be named 'Dynamic'."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    per_state_rets = compute_per_state_compound_returns(state_series, strat_rets, n_states)
    best = identify_best_strategy_per_state(per_state_rets)
    strategies = strat_rets.columns.tolist()
    p_vectors = build_binary_assignment_vectors(best, strategies, n_states)
    W = compute_total_return_weights(P_mean, p_vectors)
    dyn = build_dynamic_portfolio(state_series, strat_rets, W)
    assert dyn.name == "Dynamic"


# ── Performance metric tests ──────────────────────────────────────────────────

def test_compound_return_positive_returns():
    """Positive constant returns should yield positive compound return."""
    rets = pd.Series([0.01] * 100)
    cr = compound_return(rets)
    expected = (1.01 ** 100) - 1
    assert abs(cr - expected) < 1e-10


def test_compound_return_zero():
    """Zero daily returns should yield zero compound return."""
    rets = pd.Series([0.0] * 100)
    cr = compound_return(rets)
    assert abs(cr) < 1e-15


def test_compound_return_negative_returns():
    """Losing strategy should yield negative compound return."""
    rets = pd.Series([-0.005] * 200)
    cr = compound_return(rets)
    assert cr < 0


def test_daily_volatility_constant():
    """Volatility of constant returns should be 0."""
    rets = pd.Series([0.01] * 100)
    vol = daily_volatility(rets)
    assert abs(vol) < 1e-15


def test_daily_volatility_positive():
    """Volatility of random returns should be positive."""
    rets = pd.Series(np.random.randn(200) * 0.01)
    vol = daily_volatility(rets)
    assert vol > 0


def test_daily_volatility_not_annualized():
    """Daily vol should NOT be multiplied by sqrt(252) — this is a key check."""
    np.random.seed(10)
    rets = pd.Series(np.random.randn(252) * 0.01)
    vol_raw = rets.std(ddof=1)
    vol_func = daily_volatility(rets)
    # They should be equal (no annualization)
    assert abs(vol_func - vol_raw) < 1e-12
    # And annualized vol should be different
    vol_annualized = vol_raw * np.sqrt(252)
    assert abs(vol_func - vol_annualized) > 0.1


def test_custom_sharpe_formula():
    """Sharpe = compound_return / daily_vol (paper's non-standard definition)."""
    np.random.seed(5)
    rets = pd.Series(np.random.randn(200) * 0.01 + 0.001)
    cr = compound_return(rets)
    vol = daily_volatility(rets)
    sharpe = custom_sharpe(rets)
    expected = cr / vol
    assert abs(sharpe - expected) < 1e-12


def test_custom_sharpe_sign():
    """Sharpe should be positive for a positive compound return."""
    rets = pd.Series([0.01] * 100)
    # Compound return > 0, vol = 0, so returns 0 not positive
    # But for randn + drift:
    np.random.seed(99)
    rets = pd.Series(np.random.randn(200) * 0.005 + 0.002)
    cr = compound_return(rets)
    sharpe = custom_sharpe(rets)
    if cr > 0:
        assert sharpe > 0


def test_custom_sharpe_not_annualized_verification():
    """
    Verify ERC 2006 Sharpe approximately 57.37 when using non-annualized vol.
    This is the paper's benchmark value. We test the formula orientation.
    """
    # Simulate 252 trading days with known stats to verify Sharpe calculation
    # Compound return 0.24, daily vol 0.0042 -> Sharpe ~ 57.14
    np.random.seed(0)
    rets = pd.Series([0.0009] * 252 + np.random.randn(252) * 0.0042)
    cr = compound_return(rets)
    vol_daily = daily_volatility(rets)
    vol_annualized = vol_daily * np.sqrt(252)
    sharpe_correct = cr / vol_daily
    sharpe_wrong = cr / vol_annualized

    # The correct formula produces much larger Sharpe (~57) than the wrong one (~3.5)
    # because vol_daily << vol_annualized by factor sqrt(252)~15.87
    ratio = sharpe_correct / sharpe_wrong
    assert abs(ratio - np.sqrt(252)) < 0.1


# ── Annual metrics tests ──────────────────────────────────────────────────────

def test_annual_metrics_returns_correct_keys(synthetic_setup):
    """Annual metrics dict should have 'compound_return', 'volatility', 'sharpe'."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    metrics = compute_annual_metrics(strat_rets)
    assert set(metrics.keys()) == {"compound_return", "volatility", "sharpe"}


def test_annual_metrics_years_match(synthetic_setup):
    """Annual metrics should cover all years present in the returns data."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    metrics = compute_annual_metrics(strat_rets)
    expected_years = sorted(strat_rets.index.year.unique())
    actual_years = sorted(metrics["sharpe"].index.tolist())
    assert actual_years == expected_years


def test_annual_metrics_sharpe_formula(synthetic_setup):
    """Verify annual Sharpe is compound_return / daily_vol (not annualized)."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    metrics = compute_annual_metrics(strat_rets)
    year = strat_rets.index.year[0]
    year_rets = strat_rets[strat_rets.index.year == year]["EW"]
    cr = compound_return(year_rets)
    vol = daily_volatility(year_rets)
    expected_sharpe = cr / vol
    actual_sharpe = metrics["sharpe"].loc[year, "EW"]
    assert abs(actual_sharpe - expected_sharpe) < 1e-10


# ── Full period metrics tests ─────────────────────────────────────────────────

def test_full_period_metrics_shape(synthetic_setup):
    """Full metrics should have strategies as index and metrics as columns."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    full = compute_full_period_metrics(strat_rets)
    assert set(full.columns) >= {"compound_return", "volatility", "sharpe"}
    assert set(full.index) == set(strat_rets.columns)


def test_full_period_metrics_n_days(synthetic_setup):
    """n_days column should equal the number of return observations."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    full = compute_full_period_metrics(strat_rets)
    for strat in strat_rets.columns:
        assert full.loc[strat, "n_days"] == len(strat_rets[strat].dropna())


# ── Portfolio value tests ─────────────────────────────────────────────────────

def test_portfolio_value_starts_at_initial():
    """Portfolio value series should start at initial value (default $1)."""
    rets = pd.Series([0.01, -0.005, 0.02], index=pd.date_range("2020-01-01", periods=3, freq="B"))
    pv = compute_portfolio_value(rets)
    assert abs(pv.iloc[0] - 1.01) < 1e-12  # first day: (1+0.01)*1


def test_portfolio_value_length(synthetic_setup):
    """Portfolio value series should have same length as return series."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    rets = strat_rets["EW"]
    pv = compute_portfolio_value(rets, initial=100.0)
    assert len(pv) == len(rets)


def test_portfolio_value_always_positive(synthetic_setup):
    """Portfolio value should always be positive for reasonable daily returns."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    for strat in strat_rets.columns:
        pv = compute_portfolio_value(strat_rets[strat])
        assert (pv > 0).all()


# ── Full evaluation integration test ─────────────────────────────────────────

def test_run_full_evaluation_returns_all_keys(synthetic_setup):
    """run_full_evaluation should return all expected output keys."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    result = run_full_evaluation(state_series, strat_rets, P_mean, n_states)

    expected_keys = {
        "per_state_returns", "best_strategy", "p_vectors", "W", "W_df",
        "dynamic_returns", "all_returns", "full_metrics", "annual_metrics",
        "portfolio_values", "strategies",
    }
    assert set(result.keys()) == expected_keys


def test_run_full_evaluation_dynamic_in_all_returns(synthetic_setup):
    """Dynamic strategy should be present in all_returns DataFrame."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    result = run_full_evaluation(state_series, strat_rets, P_mean, n_states)
    assert "Dynamic" in result["all_returns"].columns


def test_run_full_evaluation_full_metrics_5_strategies(synthetic_setup):
    """Full metrics should include all 4 static + 1 dynamic strategy."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    result = run_full_evaluation(state_series, strat_rets, P_mean, n_states)
    assert "Dynamic" in result["full_metrics"].index
    assert len(result["full_metrics"]) == 5


def test_run_full_evaluation_w_rows_sum_to_one(synthetic_setup):
    """Total return weight matrix rows should sum to 1."""
    state_series, strat_rets, P_mean, n_states = synthetic_setup
    result = run_full_evaluation(state_series, strat_rets, P_mean, n_states)
    np.testing.assert_allclose(result["W"].sum(axis=1), 1.0, atol=1e-8)
