"""
Static portfolio weight computation.

Computes Equal Weight, Minimum Variance, Maximum Diversification, and Equal
Risk Contribution portfolios using the full-period sample covariance matrix.
Optimization uses scipy.optimize.minimize with SLSQP method.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize  # Custom optimization — Context7 confirmed scipy SLSQP


# ── Objective functions ──────────────────────────────────────────────────────

def _portfolio_variance(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Compute portfolio variance w'Σw."""
    return float(w @ Sigma @ w)


def _neg_diversification_ratio(w: np.ndarray, Sigma: np.ndarray, sigma_vec: np.ndarray) -> float:
    """
    Negative diversification ratio for maximization.

    DR(w) = (Σ_i w_i * σ_i) / sqrt(w'Σw)
    """
    port_vol = np.sqrt(w @ Sigma @ w)
    weighted_vols = np.dot(w, sigma_vec)
    if port_vol < 1e-12:
        return 0.0
    return -weighted_vols / port_vol


def _erc_objective(w: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Equal Risk Contribution objective.

    Minimize sum_i (TRC_i - mean(TRC))^2 where TRC_i = w_i * (Σw)_i.
    Custom — Context7 found no library equivalent (paper ERC formulation).
    """
    Sigma_w = Sigma @ w
    TRC = w * Sigma_w
    mean_TRC = np.mean(TRC)
    return float(np.sum((TRC - mean_TRC) ** 2))


# ── Portfolio optimizers ─────────────────────────────────────────────────────

def compute_equal_weight(n: int) -> np.ndarray:
    """
    Compute equal-weight portfolio: w_i = 1/n for all i.

    Parameters
    ----------
    n : int
        Number of assets.

    Returns
    -------
    np.ndarray
        Weight vector of shape (n,).
    """
    return np.ones(n) / n


def compute_min_variance(returns: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    """
    Solve the Minimum Variance portfolio optimization.

    Minimize w'Σw subject to: sum(w) = 1, w_i >= 0.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for all assets (full period).

    Returns
    -------
    w_opt : np.ndarray
        Optimal weight vector.
    success : bool
        Whether the optimizer converged.
    """
    n = returns.shape[1]
    Sigma_np = returns.cov().values
    # Regularize if near-singular
    Sigma_np = Sigma_np + 1e-8 * np.eye(n)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        _portfolio_variance,
        x0,
        args=(Sigma_np,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    w_opt = result.x
    w_opt = np.clip(w_opt, 0.0, 1.0)
    w_opt /= w_opt.sum()  # re-normalize

    return w_opt, result.success


def compute_max_diversification(returns: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    """
    Solve the Maximum Diversification portfolio optimization.

    Maximize DR(w) = (Σ_i w_i σ_i) / sqrt(w'Σw) subject to: sum(w)=1, w_i>=0.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for all assets (full period).

    Returns
    -------
    w_opt : np.ndarray
        Optimal weight vector.
    success : bool
        Whether the optimizer converged.
    """
    n = returns.shape[1]
    Sigma_np = returns.cov().values
    Sigma_np = Sigma_np + 1e-8 * np.eye(n)
    sigma_vec = np.sqrt(np.diag(Sigma_np))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    best_result = None
    best_val = np.inf

    # Multiple random starting points for robustness
    rng = np.random.default_rng(42)
    starting_points = [np.ones(n) / n]
    for _ in range(5):
        rand_w = rng.dirichlet(np.ones(n))
        starting_points.append(rand_w)

    for x0 in starting_points:
        result = minimize(
            _neg_diversification_ratio,
            x0,
            args=(Sigma_np, sigma_vec),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.fun < best_val:
            best_val = result.fun
            best_result = result

    w_opt = best_result.x
    w_opt = np.clip(w_opt, 0.0, 1.0)
    w_opt /= w_opt.sum()

    return w_opt, best_result.success


def compute_erc(returns: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    """
    Solve the Equal Risk Contribution (ERC) portfolio optimization.

    Minimize sum_i (TRC_i - mean(TRC))^2 subject to: sum(w)=1, w_i>=0.
    Uses multiple starting points for robust convergence.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for all assets (full period).

    Returns
    -------
    w_opt : np.ndarray
        Optimal weight vector.
    success : bool
        Whether the optimizer converged.
    """
    n = returns.shape[1]
    Sigma_np = returns.cov().values
    Sigma_np = Sigma_np + 1e-8 * np.eye(n)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n  # small positive lower bound for numerical stability

    # Inverse-vol starting point (good warm start for ERC)
    sigma_diag = np.sqrt(np.diag(Sigma_np))
    x0_inv_vol = (1.0 / sigma_diag) / (1.0 / sigma_diag).sum()

    best_result = None
    best_val = np.inf

    rng = np.random.default_rng(42)
    starting_points = [np.ones(n) / n, x0_inv_vol]
    for _ in range(8):
        rand_w = rng.dirichlet(np.ones(n))
        starting_points.append(rand_w)

    for x0 in starting_points:
        result = minimize(
            _erc_objective,
            x0,
            args=(Sigma_np,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 5000, "ftol": 1e-15},
        )
        if result.fun < best_val:
            best_val = result.fun
            best_result = result

    w_opt = best_result.x
    w_opt = np.clip(w_opt, 0.0, 1.0)
    w_opt /= w_opt.sum()

    return w_opt, best_result.success


def validate_weights(w: np.ndarray, name: str, returns: pd.DataFrame) -> None:
    """
    Validate that portfolio weights are feasible and optimizer converged.

    Parameters
    ----------
    w : np.ndarray
        Portfolio weight vector.
    name : str
        Strategy name for display.
    returns : pd.DataFrame
        Returns used to compute ERC convergence check.
    """
    assert abs(w.sum() - 1.0) < 1e-6, f"{name}: weights do not sum to 1. Sum = {w.sum():.8f}"
    assert np.all(w >= -1e-8), f"{name}: negative weights found: {w.min():.6f}"
    print(f"  {name}: sum={w.sum():.8f}, min={w.min():.6f}, max={w.max():.6f}")

    if name == "ERC":
        Sigma_np = returns.cov().values + 1e-8 * np.eye(len(w))
        trc = w * (Sigma_np @ w)
        trc_range = trc.max() - trc.min()
        print(f"  ERC TRC range: {trc_range:.2e} (expect < 1e-6)")


def compute_strategy_returns(
    returns: pd.DataFrame,
    weights: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute daily return series for each static portfolio strategy.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns.
    weights : dict
        Mapping of strategy name to weight vector.

    Returns
    -------
    pd.DataFrame
        Strategy return series as columns, dates as index.
    """
    strategy_returns = {}
    for name, w in weights.items():
        strategy_returns[name] = (returns * w).sum(axis=1)
    return pd.DataFrame(strategy_returns)


def compute_all_static_portfolios(
    returns: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Compute all four static portfolio strategies for a given returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Full-period daily returns (all assets).

    Returns
    -------
    weights : dict
        Strategy name -> weight array.
    strategy_returns : pd.DataFrame
        Daily return series for each strategy.
    """
    n = returns.shape[1]

    print("\n── Computing Static Portfolio Weights ──────────────")

    # Equal Weight
    w_ew = compute_equal_weight(n)
    print(f"  EW: all weights = {1/n:.6f}")

    # Minimum Variance
    w_mv, success_mv = compute_min_variance(returns)
    print(f"  MinVar convergence: {success_mv}")

    # Maximum Diversification
    w_md, success_md = compute_max_diversification(returns)
    print(f"  MaxDiv convergence: {success_md}")

    # Equal Risk Contribution
    w_erc, success_erc = compute_erc(returns)
    print(f"  ERC convergence: {success_erc}")

    weights = {
        "EW": w_ew,
        "MinVar": w_mv,
        "MaxDiv": w_md,
        "ERC": w_erc,
    }

    print("\n── Validating Weights ──────────────────────────────")
    for name, w in weights.items():
        validate_weights(w, name, returns)

    strategy_rets = compute_strategy_returns(returns, weights)
    return weights, strategy_rets
