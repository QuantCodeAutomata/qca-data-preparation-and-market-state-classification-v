"""
Market state classification via K-means clustering of rolling volatility.

Applies K-means (k=10) to the 22-day rolling portfolio volatility to identify
10 discrete market regimes ordered from lowest to highest volatility.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans  # Using sklearn KMeans — Context7 confirmed


def fit_kmeans_states(
    rolling_vol: pd.Series,
    n_clusters: int = 10,
    n_init: int = 50,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, KMeans]:
    """
    Apply K-means clustering to rolling volatility to identify market regimes.

    Clusters are ordered by centroid magnitude: state 1 = lowest volatility,
    state 10 = highest volatility.

    Parameters
    ----------
    rolling_vol : pd.Series
        22-day rolling portfolio volatility (NaN values dropped internally).
    n_clusters : int
        Number of market states (default 10).
    n_init : int
        Number of K-means random initializations (default 50 for stability).
    max_iter : int
        Maximum K-means iterations per run (default 1000).
    random_state : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    states : np.ndarray
        Ordered state labels (1 to n_clusters) for each valid date.
    centroids_ordered : np.ndarray
        Ordered centroid values (shape: n_clusters).
    valid_vol : np.ndarray
        Valid volatility values (NaN removed) used for clustering.
    km : KMeans
        Fitted KMeans object.
    """
    # Drop NaN values (first 21 rows from rolling window warmup)
    valid_vol = rolling_vol.dropna().values  # shape: (T-21,)

    # Reshape to 2D for sklearn: (n_samples, 1)
    X = valid_vol.reshape(-1, 1)

    # Using sklearn KMeans — Context7 confirmed
    km = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    km.fit(X)

    # ── State ordering: map raw labels to ordered labels ──────────────────
    # centroids shape: (n_clusters,)
    centroids = km.cluster_centers_.flatten()

    # order[i] = original KMeans label that has the i-th smallest centroid
    order = np.argsort(centroids)

    # mapping: original_label -> ordered_state (1-indexed)
    # Custom — Context7 found no library equivalent (paper Eq. state ordering)
    mapping = {order[i]: i + 1 for i in range(n_clusters)}
    states = np.array([mapping[label] for label in km.labels_])

    # Ordered centroids (ascending)
    centroids_ordered = centroids[order]

    return states, centroids_ordered, valid_vol, km


def build_state_series(
    rolling_vol: pd.Series,
    states: np.ndarray,
) -> pd.Series:
    """
    Construct a pandas Series of state labels indexed by trading date.

    Parameters
    ----------
    rolling_vol : pd.Series
        Full rolling volatility series (with NaN for warmup period).
    states : np.ndarray
        Ordered state labels from K-means (length = number of valid vol values).

    Returns
    -------
    pd.Series
        State labels indexed by date, starting from the first valid vol date.
    """
    valid_dates = rolling_vol.dropna().index
    assert len(valid_dates) == len(states), (
        f"Length mismatch: {len(valid_dates)} dates vs {len(states)} states"
    )
    state_series = pd.Series(states, index=valid_dates, name="MarketState", dtype=int)
    return state_series


def validate_state_distribution(state_series: pd.Series, n_clusters: int = 10) -> dict:
    """
    Validate that the majority of trading days fall in low-volatility states.

    Parameters
    ----------
    state_series : pd.Series
        State label series.
    n_clusters : int
        Total number of states.

    Returns
    -------
    dict
        State count distribution.
    """
    counts = state_series.value_counts().sort_index()
    total = len(state_series)

    print("\n── State Distribution ──────────────────────────────")
    for state in range(1, n_clusters + 1):
        count = counts.get(state, 0)
        pct = count / total * 100
        print(f"  State {state:2d}: {count:5d} days ({pct:5.1f}%)")

    low_vol_pct = sum(counts.get(s, 0) for s in range(1, 4)) / total * 100
    print(f"\n  States 1-3 total: {low_vol_pct:.1f}% (expect >60% for equity-only sets)")
    # Asset Set 2 includes BTC-USD whose higher baseline volatility spreads
    # observations across more states, so the >60% threshold may not hold there.
    if low_vol_pct <= 30:
        print(f"  WARNING: States 1-3 share ({low_vol_pct:.1f}%) is low; "
              "likely due to high-volatility assets (e.g. BTC-USD).")

    return counts.to_dict()
