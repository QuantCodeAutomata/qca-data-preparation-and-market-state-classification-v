"""
Bayesian Markov Transition Matrix Construction.

Counts state transitions, applies uniform Dirichlet priors (alpha=1),
computes posterior mean transition probabilities, validates convergence via
Gelman-Rubin diagnostics, and computes SLEM and mixing time bounds.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ── Transition count matrix ──────────────────────────────────────────────────

def count_transitions(state_series: pd.Series, n_states: int = 10) -> np.ndarray:
    """
    Count observed state-to-state transitions.

    Parameters
    ----------
    state_series : pd.Series
        Chronologically ordered state labels (integers 1 to n_states).
    n_states : int
        Number of states (default 10).

    Returns
    -------
    N : np.ndarray
        Transition count matrix of shape (n_states, n_states).
        N[i, j] = number of transitions from state i+1 to state j+1.
    """
    states = state_series.values
    N = np.zeros((n_states, n_states), dtype=int)

    for t in range(len(states) - 1):
        i = states[t] - 1      # 0-indexed from state
        j = states[t + 1] - 1  # 0-indexed to state
        N[i, j] += 1

    total_transitions = N.sum()
    print(f"\n── Transition Count Matrix ─────────────────────────")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Diagonal sum: {np.diag(N).sum()} ({np.diag(N).sum()/total_transitions*100:.1f}%)")
    return N


def compute_posterior_mean(N: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Compute posterior mean transition matrix using uniform Dirichlet prior.

    Posterior: P_{ij} ~ Dirichlet(alpha + N[i,:]) for each row i.
    Posterior mean: E[P_{ij}] = (alpha + N[i,j]) / (n*alpha + sum_k N[i,k])

    Parameters
    ----------
    N : np.ndarray
        Transition count matrix (n_states x n_states).
    alpha : float
        Dirichlet concentration parameter (uniform prior, default 1.0).

    Returns
    -------
    P_mean : np.ndarray
        Posterior mean transition matrix, rows sum to 1.
    """
    n_states = N.shape[0]
    # Custom — Dirichlet posterior mean computation (paper specification)
    alpha_matrix = alpha * np.ones((n_states, n_states))
    posterior_counts = alpha_matrix + N
    row_sums = posterior_counts.sum(axis=1, keepdims=True)
    P_mean = posterior_counts / row_sums

    # Validate rows sum to 1
    assert np.allclose(P_mean.sum(axis=1), 1.0, atol=1e-10), "Rows of P_mean do not sum to 1"
    return P_mean


# ── Gibbs sampling ───────────────────────────────────────────────────────────

def gibbs_sampling(
    N: np.ndarray,
    alpha: float = 1.0,
    n_chains: int = 4,
    n_iter: int = 10000,
    burn_in: int = 2000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Gibbs sampling for Bayesian Markov transition matrix.

    Since the posterior is conjugate (Dirichlet), each Gibbs step produces
    an exact independent draw from the posterior. Convergence is trivially fast.

    Parameters
    ----------
    N : np.ndarray
        Transition count matrix (n_states x n_states).
    alpha : float
        Dirichlet prior concentration parameter.
    n_chains : int
        Number of parallel chains for Gelman-Rubin diagnostics.
    n_iter : int
        Total iterations per chain.
    burn_in : int
        Burn-in iterations to discard.
    random_state : int
        Random seed.

    Returns
    -------
    samples : np.ndarray
        Post-burn-in samples, shape (n_chains, n_post_burnin, n_states, n_states).
    """
    n_states = N.shape[0]
    n_post_burnin = n_iter - burn_in
    rng = np.random.default_rng(random_state)

    # Posterior concentration parameters: same for all chains (conjugate)
    concentration = alpha * np.ones((n_states, n_states)) + N  # shape: (n_states, n_states)

    samples = np.zeros((n_chains, n_post_burnin, n_states, n_states))

    for chain in range(n_chains):
        for it in range(n_iter):
            # Sample each row independently from its Dirichlet posterior
            P_sample = np.zeros((n_states, n_states))
            for i in range(n_states):
                P_sample[i, :] = rng.dirichlet(concentration[i, :])

            if it >= burn_in:
                samples[chain, it - burn_in, :, :] = P_sample

    return samples


def gelman_rubin_rhat(samples: np.ndarray) -> np.ndarray:
    """
    Compute Gelman-Rubin R-hat for each parameter.

    Custom implementation — uses standard Gelman-Rubin formula.

    Parameters
    ----------
    samples : np.ndarray
        Shape (n_chains, n_post_burnin, n_states, n_states).

    Returns
    -------
    rhat : np.ndarray
        R-hat values, shape (n_states, n_states).
    """
    n_chains, m, n_states, _ = samples.shape  # m = post-burnin chain length

    rhat = np.zeros((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            chains_ij = samples[:, :, i, j]  # shape: (n_chains, m)

            chain_means = chains_ij.mean(axis=1)        # (n_chains,)
            grand_mean = chain_means.mean()

            # Between-chain variance
            B = m * np.var(chain_means, ddof=1)

            # Within-chain variance
            W = np.var(chains_ij, axis=1, ddof=1).mean()

            if W < 1e-15:
                rhat[i, j] = 1.0  # degenerate case; trivially converged
                continue

            # Pooled variance estimate
            V_hat = ((m - 1) / m) * W + ((n_chains + 1) / (n_chains * m)) * B
            rhat[i, j] = np.sqrt(V_hat / W)

    return rhat


# ── Eigenspectrum and mixing times ───────────────────────────────────────────

def compute_eigenspectrum(P_mean: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute eigenspectrum of the transition matrix.

    Parameters
    ----------
    P_mean : np.ndarray
        Posterior mean transition matrix (row-stochastic).

    Returns
    -------
    slem : float
        Second-largest eigenvalue modulus.
    lambda_1 : float
        Largest eigenvalue (should be ~1.0).
    eigenvalues : np.ndarray
        All eigenvalues sorted by descending modulus.
    """
    eigenvalues = np.linalg.eigvals(P_mean)
    abs_eigenvalues = np.abs(eigenvalues)
    sorted_idx = np.argsort(abs_eigenvalues)[::-1]  # descending
    sorted_eigenvalues = eigenvalues[sorted_idx]
    sorted_abs = abs_eigenvalues[sorted_idx]

    lambda_1 = sorted_abs[0]
    slem = sorted_abs[1]

    assert abs(lambda_1 - 1.0) < 1e-4, f"Largest eigenvalue should be ~1. Got {lambda_1:.6f}"
    return slem, float(lambda_1), sorted_eigenvalues


def compute_stationary_distribution(P_mean: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution pi of the Markov chain.

    Solves pi * P = pi (left eigenvector for eigenvalue 1).

    Parameters
    ----------
    P_mean : np.ndarray
        Posterior mean transition matrix.

    Returns
    -------
    pi : np.ndarray
        Stationary distribution (sums to 1, all positive).
    """
    # Left eigenvectors of P_mean = right eigenvectors of P_mean.T
    eigenvalues, eigenvectors = np.linalg.eig(P_mean.T)
    # Find eigenvector corresponding to eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()  # normalize to sum to 1

    assert np.allclose(pi.sum(), 1.0, atol=1e-8), f"Stationary dist does not sum to 1: {pi.sum()}"
    assert np.all(pi > -1e-8), f"Negative stationary probabilities found"
    pi = np.abs(pi)  # ensure non-negative
    pi /= pi.sum()
    return pi


def compute_mixing_times(
    slem: float,
    pi: np.ndarray,
    epsilon: float = 0.01,
) -> Dict[str, float]:
    """
    Compute mixing time bounds for the Markov chain.

    Based on spectral gap and stationary distribution.

    Parameters
    ----------
    slem : float
        Second-largest eigenvalue modulus (lambda_2).
    pi : np.ndarray
        Stationary distribution.
    epsilon : float
        Distance threshold (default 0.01).

    Returns
    -------
    dict with keys: t_rel, t_mix_lower, t_mix_upper
    """
    # Custom — no library equivalent (paper mixing time formulas)
    spectral_gap = 1.0 - slem
    t_rel = 1.0 / spectral_gap

    pi_min = np.min(pi)
    log_inv_eps = np.log(1.0 / (2.0 * epsilon))

    t_mix_lower = (t_rel - 1.0) * log_inv_eps
    t_mix_upper = t_rel * (0.5 * np.log(1.0 / pi_min) + log_inv_eps)

    return {
        "slem": slem,
        "spectral_gap": spectral_gap,
        "t_rel": t_rel,
        "t_mix_lower": t_mix_lower,
        "t_mix_upper": t_mix_upper,
        "epsilon": epsilon,
    }


def build_transition_matrix(
    state_series: pd.Series,
    n_states: int = 10,
    alpha: float = 1.0,
    n_chains: int = 4,
    n_iter: int = 10000,
    burn_in: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Full pipeline: count transitions, compute posterior mean, run Gibbs sampling,
    compute R-hat diagnostics, eigenspectrum, and mixing times.

    Parameters
    ----------
    state_series : pd.Series
        Chronologically ordered state labels.
    n_states : int
        Number of states (default 10).
    alpha : float
        Dirichlet prior concentration (default 1.0, uniform).
    n_chains, n_iter, burn_in : Gibbs sampling parameters.

    Returns
    -------
    P_mean : np.ndarray  (n_states x n_states)
    N : np.ndarray       (n_states x n_states) transition counts
    rhat : np.ndarray    (n_states x n_states) Gelman-Rubin R-hat
    diagnostics : dict   mixing times, SLEM, etc.
    """
    N = count_transitions(state_series, n_states)
    P_mean = compute_posterior_mean(N, alpha)

    print("\n── Gibbs Sampling ──────────────────────────────────")
    samples = gibbs_sampling(N, alpha, n_chains, n_iter, burn_in)

    # Validate posterior mean from Gibbs matches analytical result
    gibbs_mean = samples.mean(axis=(0, 1))  # average over chains and iterations
    max_diff = np.max(np.abs(gibbs_mean - P_mean))
    print(f"  Max |Gibbs mean - Analytical mean|: {max_diff:.6f} (expect < 0.01)")

    print("\n── Gelman-Rubin Diagnostics ────────────────────────")
    rhat = gelman_rubin_rhat(samples)
    max_rhat = rhat.max()
    print(f"  Max R-hat: {max_rhat:.4f} (expect <= 1.1)")
    assert max_rhat <= 1.1, f"Gelman-Rubin R-hat exceeds 1.1: {max_rhat:.4f}"

    print("\n── Eigenspectrum Analysis ──────────────────────────")
    slem, lambda_1, eigenvalues = compute_eigenspectrum(P_mean)
    print(f"  Lambda_1 (largest): {lambda_1:.6f}")
    print(f"  SLEM (lambda_2):    {slem:.6f}")

    pi = compute_stationary_distribution(P_mean)
    mixing = compute_mixing_times(slem, pi, epsilon=0.01)

    print(f"  t_rel = {mixing['t_rel']:.2f}")
    print(f"  t_mix lower bound: {mixing['t_mix_lower']:.2f}")
    print(f"  t_mix upper bound: {mixing['t_mix_upper']:.2f}")

    diagnostics = {
        **mixing,
        "stationary_dist": pi,
        "lambda_1": lambda_1,
        "max_rhat": max_rhat,
        "gibbs_max_diff": max_diff,
        "eigenvalues": eigenvalues,
    }

    return P_mean, N, rhat, diagnostics
