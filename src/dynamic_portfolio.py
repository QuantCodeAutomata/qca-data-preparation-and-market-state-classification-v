"""
Dynamic Portfolio Construction via Markov-Weighted Strategy Blending.

Identifies the best static strategy per market state, builds binary assignment
vectors, computes total return weights via Markov matrix-vector products, and
generates the dynamic portfolio daily return series.

Also evaluates all strategies on annual and full-period performance metrics
using the paper's non-standard Sharpe ratio definition.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Per-state strategy evaluation ────────────────────────────────────────────

def compute_per_state_compound_returns(
    state_series: pd.Series,
    strategy_returns: pd.DataFrame,
    n_states: int = 10,
) -> pd.DataFrame:
    """
    Compute cumulative compound return per strategy per market state.

    For each state s, collects all days (not necessarily consecutive) where the
    market was in state s, and computes prod(1 + r_t) - 1 for each strategy.

    Parameters
    ----------
    state_series : pd.Series
        State label series indexed by date.
    strategy_returns : pd.DataFrame
        Daily returns for each static strategy (columns: EW, MinVar, MaxDiv, ERC).
    n_states : int
        Number of market states.

    Returns
    -------
    pd.DataFrame
        Compound returns, shape (n_states, n_strategies).
        Index: state labels 1..n_states, columns: strategy names.
    """
    strategies = strategy_returns.columns.tolist()
    results = {}

    for state in range(1, n_states + 1):
        state_dates = state_series[state_series == state].index
        # Intersect with available strategy return dates
        valid_dates = state_dates.intersection(strategy_returns.index)

        state_row = {}
        for strat in strategies:
            rets = strategy_returns.loc[valid_dates, strat]
            cum_ret = (1.0 + rets).prod() - 1.0
            state_row[strat] = cum_ret

        results[state] = state_row

    df = pd.DataFrame(results).T  # shape: (n_states, n_strategies)
    df.index.name = "State"
    return df


def identify_best_strategy_per_state(
    per_state_returns: pd.DataFrame,
) -> pd.Series:
    """
    Identify the best-performing strategy in each market state.

    Parameters
    ----------
    per_state_returns : pd.DataFrame
        Compound returns per state per strategy.

    Returns
    -------
    pd.Series
        Best strategy name for each state (index: state label 1..10).
    """
    best = per_state_returns.idxmax(axis=1)
    best.name = "BestStrategy"
    return best


def build_binary_assignment_vectors(
    best_strategy: pd.Series,
    strategies: List[str],
    n_states: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Build binary assignment vectors p_m for each strategy.

    p_m[s] = 1 if strategy m is the best in state s, else 0.

    Parameters
    ----------
    best_strategy : pd.Series
        Best strategy name for each state (index: 1..n_states).
    strategies : list of str
        Strategy names.
    n_states : int
        Number of states.

    Returns
    -------
    dict mapping strategy name to binary vector of shape (n_states,).
    """
    p_vectors = {}
    for m in strategies:
        p_m = np.zeros(n_states, dtype=float)
        for s in range(1, n_states + 1):
            if best_strategy.get(s) == m:
                p_m[s - 1] = 1.0
        p_vectors[m] = p_m

    # Validate mutual exclusivity: exactly one strategy is best per state
    total = sum(p_vectors[m] for m in strategies)
    assert np.all(total == 1.0), f"Not exactly one best strategy per state: {total}"

    return p_vectors


def compute_total_return_weights(
    P_mean: np.ndarray,
    p_vectors: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute total return weight matrix W = [P @ p_m for each strategy m].

    W[s, m] = probability-weighted fraction of state-s returns attributed
              to strategy m (via Markov blending).

    Parameters
    ----------
    P_mean : np.ndarray
        Bayesian posterior mean transition matrix (n_states x n_states).
    p_vectors : dict
        Binary assignment vectors per strategy.

    Returns
    -------
    W : np.ndarray
        Weight matrix, shape (n_states, n_strategies).
        Each row sums to 1.0.
    """
    strategies = list(p_vectors.keys())
    n_states = P_mean.shape[0]
    W = np.zeros((n_states, len(strategies)))

    for idx, m in enumerate(strategies):
        W[:, idx] = P_mean @ p_vectors[m]

    # Validate: each row sums to 1 (since sum of all p_m = ones vector)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-8), (
        f"Total return weight rows do not sum to 1. Row sums: {W.sum(axis=1)}"
    )
    return W


# ── Dynamic portfolio return series ──────────────────────────────────────────

def build_dynamic_portfolio(
    state_series: pd.Series,
    strategy_returns: pd.DataFrame,
    W: np.ndarray,
) -> pd.Series:
    """
    Construct dynamic portfolio daily return series.

    At each day t, the weight vector W[state_t - 1, :] is used to blend
    strategy returns at t+1 (state today determines tomorrow's allocation).

    Parameters
    ----------
    state_series : pd.Series
        Market state labels indexed by date.
    strategy_returns : pd.DataFrame
        Daily returns for all static strategies.
    W : np.ndarray
        Total return weight matrix (n_states x n_strategies).

    Returns
    -------
    pd.Series
        Dynamic portfolio return series indexed by date (t+1 dates).
    """
    # Align state series and strategy returns by common dates
    common_dates = state_series.index.intersection(strategy_returns.index)
    state_aligned = state_series.loc[common_dates]
    strat_aligned = strategy_returns.loc[common_dates]

    dynamic_returns = []
    dynamic_dates = []

    for i in range(len(state_aligned) - 1):
        state_today = state_aligned.iloc[i]
        state_idx = int(state_today) - 1  # 0-indexed
        weights_today = W[state_idx, :]   # (n_strategies,)

        # Tomorrow's strategy returns
        tomorrow_rets = strat_aligned.iloc[i + 1].values  # (n_strategies,)

        dyn_ret = np.dot(weights_today, tomorrow_rets)
        dynamic_returns.append(dyn_ret)
        dynamic_dates.append(state_aligned.index[i + 1])

    dyn_series = pd.Series(dynamic_returns, index=pd.DatetimeIndex(dynamic_dates), name="Dynamic")
    return dyn_series


# ── Performance metrics ──────────────────────────────────────────────────────

def compound_return(returns: pd.Series) -> float:
    """Compute full-period compound return: prod(1 + r_t) - 1."""
    return float((1.0 + returns).prod() - 1.0)


def daily_volatility(returns: pd.Series) -> float:
    """Compute raw daily standard deviation (ddof=1, NOT annualized)."""
    return float(returns.std(ddof=1))


def custom_sharpe(returns: pd.Series) -> float:
    """
    Compute NON-STANDARD Sharpe ratio as defined in the paper.

    sharpe = compound_return / daily_vol (NOT annualized)

    This is a non-standard definition: numerator is cumulative compound
    return over the period, denominator is raw daily standard deviation.

    Custom — no standard library implements this exact definition.
    """
    cr = compound_return(returns)
    vol = daily_volatility(returns)
    if abs(vol) < 1e-15:
        return 0.0
    return cr / vol


def compute_annual_metrics(
    all_returns: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Compute annual performance metrics for all strategies.

    Parameters
    ----------
    all_returns : pd.DataFrame
        Daily returns for all strategies (columns), indexed by date.

    Returns
    -------
    dict with keys 'compound_return', 'volatility', 'sharpe'.
    Each value is a DataFrame with years as index and strategies as columns.
    """
    years = sorted(all_returns.index.year.unique())
    strategies = all_returns.columns.tolist()

    annual_return_rows = []
    annual_vol_rows = []
    annual_sharpe_rows = []

    for year in years:
        mask = all_returns.index.year == year
        year_rets = all_returns[mask]

        if len(year_rets) < 5:  # skip years with very few data points
            continue

        ret_row = {}
        vol_row = {}
        sharpe_row = {}
        for strat in strategies:
            s = year_rets[strat].dropna()
            if len(s) < 2:
                ret_row[strat] = np.nan
                vol_row[strat] = np.nan
                sharpe_row[strat] = np.nan
            else:
                ret_row[strat] = compound_return(s)
                vol_row[strat] = daily_volatility(s)
                sharpe_row[strat] = custom_sharpe(s)

        annual_return_rows.append((year, ret_row))
        annual_vol_rows.append((year, vol_row))
        annual_sharpe_rows.append((year, sharpe_row))

    def _to_df(rows):
        idx = [r[0] for r in rows]
        data = [r[1] for r in rows]
        return pd.DataFrame(data, index=idx)

    return {
        "compound_return": _to_df(annual_return_rows),
        "volatility": _to_df(annual_vol_rows),
        "sharpe": _to_df(annual_sharpe_rows),
    }


def compute_full_period_metrics(all_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute full-period performance metrics for all strategies.

    Parameters
    ----------
    all_returns : pd.DataFrame
        Daily returns for all strategies.

    Returns
    -------
    pd.DataFrame
        Metrics DataFrame with strategies as columns and metrics as rows.
    """
    strategies = all_returns.columns.tolist()
    rows = {}
    for strat in strategies:
        s = all_returns[strat].dropna()
        rows[strat] = {
            "compound_return": compound_return(s),
            "volatility": daily_volatility(s),
            "sharpe": custom_sharpe(s),
            "n_days": len(s),
        }
    df = pd.DataFrame(rows).T
    df.index.name = "Strategy"
    return df


def compute_portfolio_value(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """
    Compute cumulative portfolio value from daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    initial : float
        Initial portfolio value (default $1).

    Returns
    -------
    pd.Series
        Portfolio value series starting at `initial`.
    """
    cum_factor = (1.0 + returns).cumprod()
    return initial * cum_factor


def run_full_evaluation(
    state_series: pd.Series,
    strategy_returns: pd.DataFrame,
    P_mean: np.ndarray,
    n_states: int = 10,
) -> Dict:
    """
    Full dynamic portfolio pipeline: state evaluation, weight computation,
    dynamic portfolio construction, and performance evaluation.

    Parameters
    ----------
    state_series : pd.Series
        Market state labels.
    strategy_returns : pd.DataFrame
        Static strategy daily returns.
    P_mean : np.ndarray
        Bayesian transition matrix.
    n_states : int
        Number of states.

    Returns
    -------
    dict with all results.
    """
    strategies = strategy_returns.columns.tolist()

    print("\n── Per-State Compound Returns ──────────────────────")
    per_state_rets = compute_per_state_compound_returns(
        state_series, strategy_returns, n_states
    )
    print(per_state_rets.round(4).to_string())

    best_strat = identify_best_strategy_per_state(per_state_rets)
    print("\n── Best Strategy Per State ─────────────────────────")
    for s in range(1, n_states + 1):
        print(f"  State {s:2d}: {best_strat[s]}")

    p_vectors = build_binary_assignment_vectors(best_strat, strategies, n_states)
    print("\n── Binary Assignment Vectors ───────────────────────")
    for m, p in p_vectors.items():
        print(f"  p_{m} = {p.astype(int).tolist()}")

    W = compute_total_return_weights(P_mean, p_vectors)
    print("\n── Total Return Weights (rows = states) ────────────")
    W_df = pd.DataFrame(W, columns=strategies, index=range(1, n_states + 1))
    W_df.index.name = "State"
    print(W_df.round(6).to_string())

    print("\n── Building Dynamic Portfolio ──────────────────────")
    dyn_series = build_dynamic_portfolio(state_series, strategy_returns, W)

    # Combine all returns
    all_rets = strategy_returns.copy()
    # Align to common dates (dynamic starts 1 day later)
    all_rets = all_rets.loc[dyn_series.index[0]:].copy()
    all_rets["Dynamic"] = dyn_series

    print("\n── Full-Period Performance ─────────────────────────")
    full_metrics = compute_full_period_metrics(all_rets)
    print(full_metrics.to_string(float_format="{:.6f}".format))

    annual_metrics = compute_annual_metrics(all_rets)

    # Portfolio value series for each strategy
    port_values = {}
    for strat in all_rets.columns:
        port_values[strat] = compute_portfolio_value(all_rets[strat].dropna())

    return {
        "per_state_returns": per_state_rets,
        "best_strategy": best_strat,
        "p_vectors": p_vectors,
        "W": W,
        "W_df": W_df,
        "dynamic_returns": dyn_series,
        "all_returns": all_rets,
        "full_metrics": full_metrics,
        "annual_metrics": annual_metrics,
        "portfolio_values": port_values,
        "strategies": strategies,
    }
