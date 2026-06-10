"""
Visualization utilities for market state and portfolio analysis.

Generates all plots required by the experiments:
- Rolling volatility colored by market state (Figures 1, 2)
- Transition matrix heatmap (Figure 3)
- Portfolio value over time (Figures 4, 6)
- Annual Sharpe ratio bar charts (Figures 5, 7)
"""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_rolling_vol_by_state(
    rolling_vol: pd.Series,
    state_series: pd.Series,
    title: str = "Rolling 22-Day Portfolio Volatility by Market State",
    filename: str = "rolling_vol_states.png",
    n_states: int = 10,
) -> None:
    """
    Scatter plot of rolling volatility colored by market state.

    Reproduces Figures 1 and 2 from the paper.

    Parameters
    ----------
    rolling_vol : pd.Series
        22-day rolling volatility.
    state_series : pd.Series
        Market state labels indexed by date.
    title : str
        Plot title.
    filename : str
        Output filename in results/ directory.
    n_states : int
        Number of states for colormap.
    """
    _ensure_results_dir()

    # Align series
    common_dates = rolling_vol.dropna().index.intersection(state_series.index)
    vol_aligned = rolling_vol.loc[common_dates]
    states_aligned = state_series.loc[common_dates]

    fig, ax = plt.subplots(figsize=(16, 5))

    cmap = plt.get_cmap("tab10", n_states)
    scatter = ax.scatter(
        vol_aligned.index,
        vol_aligned.values,
        c=states_aligned.values,
        cmap=cmap,
        vmin=0.5,
        vmax=n_states + 0.5,
        s=5,
        alpha=0.8,
    )
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(1, n_states + 1))
    cbar.set_label("Market State")

    ax.set_xlabel("Date")
    ax.set_ylabel("22-Day Rolling Volatility")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / filename}")


def plot_transition_matrix_heatmap(
    P_mean: np.ndarray,
    title: str = "Bayesian Markov Transition Matrix",
    filename: str = "transition_matrix_heatmap.png",
) -> None:
    """
    Heatmap of the posterior mean transition matrix.

    Reproduces Figure 3 from the paper.

    Parameters
    ----------
    P_mean : np.ndarray
        10x10 posterior mean transition matrix.
    title : str
        Plot title.
    filename : str
        Output filename.
    """
    _ensure_results_dir()
    n = P_mean.shape[0]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        P_mean,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".4f",
        xticklabels=[f"S{i}" for i in range(1, n + 1)],
        yticklabels=[f"S{i}" for i in range(1, n + 1)],
        linewidths=0.3,
        cbar_kws={"label": "Transition Probability"},
    )
    ax.set_title(title)
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / filename}")


def plot_portfolio_value(
    portfolio_values: Dict[str, pd.Series],
    title: str = "Portfolio Growth ($1 Initial Investment)",
    filename: str = "portfolio_value.png",
    log_scale: bool = False,
) -> None:
    """
    Plot cumulative portfolio value for all strategies.

    Reproduces Figures 4 and 6 from the paper.

    Parameters
    ----------
    portfolio_values : dict
        Strategy name -> cumulative value Series.
    title : str
        Plot title.
    filename : str
        Output filename.
    log_scale : bool
        Use log scale on y-axis.
    """
    _ensure_results_dir()

    strategy_colors = {
        "EW": "#1f77b4",
        "MinVar": "#2ca02c",
        "MaxDiv": "#ff7f0e",
        "ERC": "#9467bd",
        "Dynamic": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    for strat, series in portfolio_values.items():
        color = strategy_colors.get(strat, None)
        lw = 2.5 if strat == "Dynamic" else 1.5
        ax.plot(series.index, series.values, label=strat, color=color, linewidth=lw)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / filename}")


def plot_annual_sharpe(
    annual_sharpe: pd.DataFrame,
    title: str = "Annual Sharpe Ratio by Strategy",
    filename: str = "annual_sharpe.png",
) -> None:
    """
    Grouped bar chart of annual Sharpe ratios.

    Reproduces Figures 5 and 7 from the paper.

    Parameters
    ----------
    annual_sharpe : pd.DataFrame
        Annual Sharpe ratios with years as index, strategies as columns.
    title : str
        Plot title.
    filename : str
        Output filename.
    """
    _ensure_results_dir()

    fig, ax = plt.subplots(figsize=(16, 6))

    strategy_colors = {
        "EW": "#1f77b4",
        "MinVar": "#2ca02c",
        "MaxDiv": "#ff7f0e",
        "ERC": "#9467bd",
        "Dynamic": "#d62728",
    }

    years = annual_sharpe.index.tolist()
    strategies = annual_sharpe.columns.tolist()
    n_strats = len(strategies)
    x = np.arange(len(years))
    width = 0.8 / n_strats

    for i, strat in enumerate(strategies):
        offsets = x + (i - n_strats / 2.0 + 0.5) * width
        values = annual_sharpe[strat].values
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=strat,
            color=strategy_colors.get(strat, None),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sharpe Ratio (compound return / daily vol)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / filename}")


def plot_centroids(
    centroids: np.ndarray,
    title: str = "K-Means Cluster Centroids by Market State",
    filename: str = "kmeans_centroids.png",
) -> None:
    """
    Bar chart of ordered K-means centroids.

    Parameters
    ----------
    centroids : np.ndarray
        Ordered centroid values for 10 states.
    title : str
    filename : str
    """
    _ensure_results_dir()

    fig, ax = plt.subplots(figsize=(8, 5))
    states = range(1, len(centroids) + 1)
    ax.bar(states, centroids * 100, color="steelblue", alpha=0.8)
    ax.set_xlabel("Market State")
    ax.set_ylabel("Centroid Volatility (%)")
    ax.set_title(title)
    ax.set_xticks(list(states))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / filename}")
