"""
Experiment 1: Data Preparation and Market State Classification via K-Means
Clustering of Rolling Volatility.

Downloads and preprocesses daily adjusted closing prices, computes rolling
volatility, and applies K-means (k=10) to identify 10 market regimes.
Saves all outputs for downstream experiments.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    ASSET_SET_1_END,
    ASSET_SET_1_START,
    ASSET_SET_2_END,
    ASSET_SET_2_START,
    get_analysis_period_data,
    load_asset_set_1,
    load_asset_set_2,
)
from src.market_states import (
    build_state_series,
    fit_kmeans_states,
    validate_state_distribution,
)
from src.visualization import plot_centroids, plot_rolling_vol_by_state

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_exp_1() -> dict:
    """
    Execute Experiment 1: Data preparation and K-means market state classification.

    Returns
    -------
    dict containing all outputs for downstream experiments.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 1: Data Preparation and Market State Classification")
    print("=" * 60)

    # ── Asset Set 1 ───────────────────────────────────────────────────────────
    print("\n── Asset Set 1 ─────────────────────────────────────────")
    prices1_full, returns1_full, eq_ret1_full, vol1_full = load_asset_set_1(cache=True)

    # Trim to analysis window (retain pre-analysis data for rolling vol warmup)
    prices1, returns1, eq_ret1, vol1 = get_analysis_period_data(
        prices1_full, returns1_full, eq_ret1_full, vol1_full,
        ASSET_SET_1_START, ASSET_SET_1_END
    )

    print(f"  Trading days in analysis window: {len(prices1)}")
    print(f"  Returns shape: {returns1.shape}")
    print(f"  Valid vol observations: {vol1.dropna().shape[0]}")

    # K-Means clustering for Asset Set 1
    states1, centroids1, valid_vol1, km1 = fit_kmeans_states(
        vol1, n_clusters=10, n_init=50, max_iter=1000, random_state=42
    )
    state_series1 = build_state_series(vol1, states1)

    print(f"\n  K-Means inertia (Asset Set 1): {km1.inertia_:.6f}")
    print(f"  Centroids (ascending, %):")
    for i, c in enumerate(centroids1):
        print(f"    State {i+1:2d}: {c*100:.4f}%")

    validate_state_distribution(state_series1)

    # ── Asset Set 2 ───────────────────────────────────────────────────────────
    print("\n── Asset Set 2 ─────────────────────────────────────────")
    prices2_full, returns2_full, eq_ret2_full, vol2_full = load_asset_set_2(cache=True)

    prices2, returns2, eq_ret2, vol2 = get_analysis_period_data(
        prices2_full, returns2_full, eq_ret2_full, vol2_full,
        ASSET_SET_2_START, ASSET_SET_2_END
    )

    print(f"  Trading days in analysis window: {len(prices2)}")
    print(f"  Returns shape: {returns2.shape}")
    print(f"  Valid vol observations: {vol2.dropna().shape[0]}")

    # K-Means clustering for Asset Set 2
    states2, centroids2, valid_vol2, km2 = fit_kmeans_states(
        vol2, n_clusters=10, n_init=50, max_iter=1000, random_state=42
    )
    state_series2 = build_state_series(vol2, states2)

    print(f"\n  K-Means inertia (Asset Set 2): {km2.inertia_:.6f}")
    print(f"  Centroids (ascending, %):")
    for i, c in enumerate(centroids2):
        print(f"    State {i+1:2d}: {c*100:.4f}%")

    validate_state_distribution(state_series2)

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n── Validation ─────────────────────────────────────────")
    print(f"  Asset Set 1 state series: {len(state_series1)} days, "
          f"from {state_series1.index[0].date()} to {state_series1.index[-1].date()}")
    print(f"  Asset Set 2 state series: {len(state_series2)} days, "
          f"from {state_series2.index[0].date()} to {state_series2.index[-1].date()}")

    # Check high-volatility states cluster around known crisis periods
    high_vol_dates_1 = state_series1[state_series1 >= 9].index
    print(f"\n  Asset Set 1 high-volatility days (states 9-10): {len(high_vol_dates_1)}")
    if len(high_vol_dates_1) > 0:
        print(f"    Date range: {high_vol_dates_1.min().date()} to {high_vol_dates_1.max().date()}")

    crisis_2008_count = sum((high_vol_dates_1.year >= 2008) & (high_vol_dates_1.year <= 2009))
    print(f"    Days in 2008-2009 crisis window: {crisis_2008_count}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n── Generating Visualizations ──────────────────────────")
    plot_rolling_vol_by_state(
        vol1, state_series1,
        title="Asset Set 1: Rolling 22-Day Volatility by Market State",
        filename="fig1_asset_set1_vol_states.png",
    )
    plot_rolling_vol_by_state(
        vol2, state_series2,
        title="Asset Set 2: Rolling 22-Day Volatility by Market State",
        filename="fig2_asset_set2_vol_states.png",
    )
    plot_centroids(
        centroids1,
        title="Asset Set 1: K-Means Centroids by State",
        filename="fig1b_asset_set1_centroids.png",
    )
    plot_centroids(
        centroids2,
        title="Asset Set 2: K-Means Centroids by State",
        filename="fig2b_asset_set2_centroids.png",
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n── Saving Outputs ─────────────────────────────────────")
    outputs = {
        # Asset Set 1
        "prices1": prices1,
        "returns1": returns1,
        "eq_ret1": eq_ret1,
        "vol1": vol1,
        "state_series1": state_series1,
        "centroids1": centroids1,
        "km1": km1,
        # Asset Set 2
        "prices2": prices2,
        "returns2": returns2,
        "eq_ret2": eq_ret2,
        "vol2": vol2,
        "state_series2": state_series2,
        "centroids2": centroids2,
        "km2": km2,
    }

    with open(DATA_DIR / "exp1_outputs.pkl", "wb") as f:
        pickle.dump(outputs, f)
    print(f"  Saved: {DATA_DIR / 'exp1_outputs.pkl'}")

    # Also save CSV files for inspection
    state_series1.to_csv(DATA_DIR / "state_series1.csv")
    state_series2.to_csv(DATA_DIR / "state_series2.csv")
    returns1.to_csv(DATA_DIR / "returns1.csv")
    returns2.to_csv(DATA_DIR / "returns2.csv")

    print("\n── Experiment 1 Complete ──────────────────────────────")
    return outputs


if __name__ == "__main__":
    run_exp_1()
