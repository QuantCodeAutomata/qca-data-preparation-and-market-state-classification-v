"""
Tests for Experiment 3: Bayesian Markov Transition Matrix Construction.

Validates:
- Transition count matrix properties
- Posterior mean computation (Dirichlet)
- Gibbs sampling shape and range
- Gelman-Rubin R-hat <= 1.1
- Eigenspectrum properties (SLEM, stationary distribution)
- Mixing time bounds
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.markov_transition import (
    build_transition_matrix,
    compute_eigenspectrum,
    compute_mixing_times,
    compute_posterior_mean,
    compute_stationary_distribution,
    count_transitions,
    gelman_rubin_rhat,
    gibbs_sampling,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_state_series():
    """Simple state series: 4 states, 500 observations."""
    np.random.seed(42)
    n_states = 4
    # Simulate a Markov chain with strong diagonal
    P = np.array([
        [0.90, 0.07, 0.02, 0.01],
        [0.05, 0.88, 0.05, 0.02],
        [0.02, 0.04, 0.90, 0.04],
        [0.01, 0.02, 0.03, 0.94],
    ])
    states = [1]
    for _ in range(499):
        s = states[-1] - 1  # 0-indexed
        next_s = np.random.choice(n_states, p=P[s]) + 1  # 1-indexed
        states.append(next_s)
    dates = pd.date_range("2020-01-02", periods=500, freq="B")
    return pd.Series(states, index=dates, name="State")


@pytest.fixture
def real_10state_series():
    """Simulated 10-state Markov chain for testing."""
    np.random.seed(0)
    n = 1000
    # Mostly stay in low states, rarely transition to high states
    states = [1]
    for _ in range(n - 1):
        s = states[-1]
        if s == 1:
            ns = np.random.choice([1, 2], p=[0.92, 0.08])
        elif s == 10:
            ns = np.random.choice([9, 10], p=[0.1, 0.9])
        else:
            ns = np.random.choice([s - 1, s, s + 1], p=[0.05, 0.90, 0.05])
        states.append(ns)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.Series(states, index=dates, name="State")


# ── Transition count matrix tests ─────────────────────────────────────────────

def test_count_transitions_shape(simple_state_series):
    """Transition matrix N should have shape (n_states, n_states)."""
    N = count_transitions(simple_state_series, n_states=4)
    assert N.shape == (4, 4)


def test_count_transitions_total(simple_state_series):
    """Total transitions should equal len(series) - 1."""
    N = count_transitions(simple_state_series, n_states=4)
    expected = len(simple_state_series) - 1
    assert N.sum() == expected


def test_count_transitions_non_negative(simple_state_series):
    """All transition counts should be non-negative."""
    N = count_transitions(simple_state_series, n_states=4)
    assert np.all(N >= 0)


def test_count_transitions_diagonal_dominant(simple_state_series):
    """Diagonal entries should be the largest in each row (regime persistence)."""
    N = count_transitions(simple_state_series, n_states=4)
    for i in range(4):
        assert N[i, i] == N[i, :].max(), (
            f"Diagonal not dominant in row {i}: diag={N[i,i]}, max={N[i,:].max()}"
        )


def test_count_transitions_integer_dtype(simple_state_series):
    """Count matrix should have integer dtype."""
    N = count_transitions(simple_state_series, n_states=4)
    assert N.dtype in [np.int32, np.int64, int]


# ── Posterior mean computation tests ─────────────────────────────────────────

def test_posterior_mean_rows_sum_to_one(simple_state_series):
    """Each row of posterior mean matrix must sum to 1.0."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)


def test_posterior_mean_non_negative(simple_state_series):
    """All posterior mean probabilities should be non-negative."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    assert np.all(P >= 0)


def test_posterior_mean_shape(simple_state_series):
    """Posterior mean matrix should have correct shape."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    assert P.shape == (4, 4)


def test_posterior_mean_formula(simple_state_series):
    """Verify posterior mean formula: E[P_ij] = (1 + N_ij) / (k + sum_k N_ik)."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)

    # Manually compute for first row
    alpha = 1.0
    n_states = 4
    expected_row0 = (alpha + N[0, :]) / (n_states * alpha + N[0, :].sum())
    np.testing.assert_allclose(P[0, :], expected_row0, rtol=1e-10)


def test_posterior_mean_uniform_prior_effect():
    """With zero observations, all probs should be 1/n (uniform Dirichlet)."""
    N = np.zeros((3, 3), dtype=int)
    P = compute_posterior_mean(N, alpha=1.0)
    expected = np.ones((3, 3)) / 3
    np.testing.assert_allclose(P, expected, atol=1e-10)


# ── Gibbs sampling tests ──────────────────────────────────────────────────────

def test_gibbs_sampling_shape(simple_state_series):
    """Gibbs samples should have shape (n_chains, n_post_burnin, n_states, n_states)."""
    N = count_transitions(simple_state_series, n_states=4)
    n_iter, burn_in = 500, 100
    samples = gibbs_sampling(N, alpha=1.0, n_chains=2, n_iter=n_iter, burn_in=burn_in)
    assert samples.shape == (2, n_iter - burn_in, 4, 4)


def test_gibbs_samples_rows_sum_to_one(simple_state_series):
    """Each sampled P matrix should have rows summing to 1."""
    N = count_transitions(simple_state_series, n_states=4)
    samples = gibbs_sampling(N, alpha=1.0, n_chains=2, n_iter=200, burn_in=50)
    # Check first chain, first sample
    row_sums = samples[0, 0, :, :].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)


def test_gibbs_samples_non_negative(simple_state_series):
    """All sampled probabilities should be non-negative."""
    N = count_transitions(simple_state_series, n_states=4)
    samples = gibbs_sampling(N, alpha=1.0, n_chains=2, n_iter=200, burn_in=50)
    assert np.all(samples >= 0)


def test_gibbs_mean_close_to_analytical(simple_state_series):
    """Gibbs posterior mean should closely match analytical posterior mean."""
    N = count_transitions(simple_state_series, n_states=4)
    P_analytical = compute_posterior_mean(N, alpha=1.0)
    samples = gibbs_sampling(N, alpha=1.0, n_chains=4, n_iter=2000, burn_in=500)
    P_gibbs_mean = samples.mean(axis=(0, 1))
    max_diff = np.max(np.abs(P_gibbs_mean - P_analytical))
    assert max_diff < 0.01, f"Max diff between Gibbs mean and analytical: {max_diff:.4f}"


# ── Gelman-Rubin diagnostics ──────────────────────────────────────────────────

def test_gelman_rubin_shape(simple_state_series):
    """R-hat array should have shape (n_states, n_states)."""
    N = count_transitions(simple_state_series, n_states=4)
    samples = gibbs_sampling(N, alpha=1.0, n_chains=4, n_iter=1000, burn_in=200)
    rhat = gelman_rubin_rhat(samples)
    assert rhat.shape == (4, 4)


def test_gelman_rubin_leq_1_1(simple_state_series):
    """All R-hat values should be <= 1.1 for conjugate Gibbs sampler."""
    N = count_transitions(simple_state_series, n_states=4)
    samples = gibbs_sampling(N, alpha=1.0, n_chains=4, n_iter=2000, burn_in=500)
    rhat = gelman_rubin_rhat(samples)
    assert rhat.max() <= 1.1, f"Max R-hat = {rhat.max():.4f}"


def test_gelman_rubin_non_negative(simple_state_series):
    """All R-hat values should be >= 0."""
    N = count_transitions(simple_state_series, n_states=4)
    samples = gibbs_sampling(N, alpha=1.0, n_chains=4, n_iter=1000, burn_in=200)
    rhat = gelman_rubin_rhat(samples)
    assert np.all(rhat >= 0)


# ── Eigenspectrum tests ───────────────────────────────────────────────────────

def test_eigenspectrum_lambda1_close_to_one(simple_state_series):
    """Largest eigenvalue of a row-stochastic matrix should be ~1."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    slem, lambda_1, _ = compute_eigenspectrum(P)
    assert abs(lambda_1 - 1.0) < 1e-4


def test_eigenspectrum_slem_less_than_one(simple_state_series):
    """SLEM should be in (0, 1) for an ergodic Markov chain."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    slem, _, _ = compute_eigenspectrum(P)
    assert 0 < slem < 1.0


def test_eigenspectrum_returns_correct_count(simple_state_series):
    """compute_eigenspectrum should return n_states eigenvalues."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    _, _, eigenvalues = compute_eigenspectrum(P)
    assert len(eigenvalues) == 4


# ── Stationary distribution tests ─────────────────────────────────────────────

def test_stationary_dist_sums_to_one(simple_state_series):
    """Stationary distribution should sum to 1."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    pi = compute_stationary_distribution(P)
    assert abs(pi.sum() - 1.0) < 1e-8


def test_stationary_dist_non_negative(simple_state_series):
    """Stationary distribution values should all be non-negative."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    pi = compute_stationary_distribution(P)
    assert np.all(pi >= 0)


def test_stationary_dist_satisfies_detailed_balance(simple_state_series):
    """pi should satisfy pi @ P = pi (stationarity condition)."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    pi = compute_stationary_distribution(P)
    pi_P = pi @ P
    np.testing.assert_allclose(pi_P, pi, atol=1e-6)


# ── Mixing time tests ─────────────────────────────────────────────────────────

def test_mixing_times_dict_keys(simple_state_series):
    """compute_mixing_times should return a dict with expected keys."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    slem, _, _ = compute_eigenspectrum(P)
    pi = compute_stationary_distribution(P)
    result = compute_mixing_times(slem, pi, epsilon=0.01)
    assert "t_rel" in result
    assert "t_mix_lower" in result
    assert "t_mix_upper" in result
    assert "slem" in result


def test_mixing_times_lower_less_than_upper(simple_state_series):
    """Lower mixing time bound should be less than upper bound."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    slem, _, _ = compute_eigenspectrum(P)
    pi = compute_stationary_distribution(P)
    result = compute_mixing_times(slem, pi, epsilon=0.01)
    assert result["t_mix_lower"] <= result["t_mix_upper"]


def test_mixing_time_t_rel_positive(simple_state_series):
    """Relaxation time t_rel should be positive."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    slem, _, _ = compute_eigenspectrum(P)
    pi = compute_stationary_distribution(P)
    result = compute_mixing_times(slem, pi, epsilon=0.01)
    assert result["t_rel"] > 0


def test_mixing_time_smaller_epsilon_larger_mixing(simple_state_series):
    """Smaller epsilon (tighter convergence) should yield larger mixing time."""
    N = count_transitions(simple_state_series, n_states=4)
    P = compute_posterior_mean(N, alpha=1.0)
    slem, _, _ = compute_eigenspectrum(P)
    pi = compute_stationary_distribution(P)
    result_strict = compute_mixing_times(slem, pi, epsilon=0.001)
    result_loose = compute_mixing_times(slem, pi, epsilon=0.1)
    assert result_strict["t_mix_lower"] > result_loose["t_mix_lower"]


# ── Full pipeline regression test ─────────────────────────────────────────────

def test_build_transition_matrix_full(real_10state_series):
    """Full pipeline: transition matrix, Gibbs, R-hat, eigenspectrum."""
    P_mean, N, rhat, diag = build_transition_matrix(
        real_10state_series,
        n_states=10,
        alpha=1.0,
        n_chains=2,
        n_iter=500,
        burn_in=100,
    )
    # Rows sum to 1
    np.testing.assert_allclose(P_mean.sum(axis=1), 1.0, atol=1e-10)
    # R-hat <= 1.1
    assert rhat.max() <= 1.1
    # SLEM in (0, 1)
    assert 0 < diag["slem"] < 1.0
    # Mixing times defined
    assert diag["t_mix_lower"] > 0
    assert diag["t_mix_upper"] > diag["t_mix_lower"]
