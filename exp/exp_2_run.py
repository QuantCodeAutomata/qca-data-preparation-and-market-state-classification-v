"""
Experiment 2: Static Portfolio Weight Computation.

Computes Equal Weight, Minimum Variance, Maximum Diversification, and Equal
Risk Contribution portfolios using full-period sample covariance matrices.
All weights are computed once and remain fixed for the entire analysis period.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio_static import compute_all_static_portfolios
from src.data_loader import (
    ASSET_SET_1_TICKERS,
    ASSET_SET_2_TICKERS,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_exp_2(exp1_outputs: dict = None) -> dict:
    """
    Execute Experiment 2: Static portfolio weight computation.

    Parameters
    ----------
    exp1_outputs : dict, optional
        Outputs from Experiment 1. If None, loads from disk.

    Returns
    -------
    dict containing all static portfolio weights and return series.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 2: Static Portfolio Weight Computation")
    print("=" * 60)

    # Load exp_1 outputs if not provided
    if exp1_outputs is None:
        with open(DATA_DIR / "exp1_outputs.pkl", "rb") as f:
            exp1_outputs = pickle.load(f)

    returns1 = exp1_outputs["returns1"]
    returns2 = exp1_outputs["returns2"]

    # ── Asset Set 1 ───────────────────────────────────────────────────────────
    print(f"\n── Asset Set 1 ({len(ASSET_SET_1_TICKERS)} assets) ──────────────────────────")
    print(f"  Using {len(returns1)} trading days for covariance estimation")

    weights1, strategy_rets1 = compute_all_static_portfolios(returns1)

    print("\n  Weight vectors (%):")
    for name, w in weights1.items():
        w_pct = [f"{wi*100:.2f}%" for wi in w]
        print(f"    {name}: {w_pct}")

    # Total compound returns (validation)
    print("\n  Full-period compound returns:")
    for strat in strategy_rets1.columns:
        total_ret = (1 + strategy_rets1[strat]).prod() - 1
        print(f"    {strat}: {total_ret:.4f} ({total_ret*100:.1f}%)")

    # ── Asset Set 2 ───────────────────────────────────────────────────────────
    print(f"\n── Asset Set 2 ({len(ASSET_SET_2_TICKERS)} assets) ──────────────────────────")
    print(f"  Using {len(returns2)} trading days for covariance estimation")

    weights2, strategy_rets2 = compute_all_static_portfolios(returns2)

    print("\n  Weight vectors (%):")
    for name, w in weights2.items():
        w_pct = [f"{wi*100:.2f}%" for wi in w]
        print(f"    {name}: {w_pct}")

    print("\n  Full-period compound returns:")
    for strat in strategy_rets2.columns:
        total_ret = (1 + strategy_rets2[strat]).prod() - 1
        print(f"    {strat}: {total_ret:.4f} ({total_ret*100:.1f}%)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n── Saving Outputs ─────────────────────────────────────")
    outputs = {
        "weights1": weights1,
        "strategy_rets1": strategy_rets1,
        "weights2": weights2,
        "strategy_rets2": strategy_rets2,
    }

    with open(DATA_DIR / "exp2_outputs.pkl", "wb") as f:
        pickle.dump(outputs, f)
    print(f"  Saved: {DATA_DIR / 'exp2_outputs.pkl'}")

    # Save weight tables as CSV
    weights1_df = pd.DataFrame(
        weights1, index=ASSET_SET_1_TICKERS
    )
    weights1_df.to_csv(DATA_DIR / "weights_asset_set1.csv")

    weights2_df = pd.DataFrame(
        weights2, index=ASSET_SET_2_TICKERS
    )
    weights2_df.to_csv(DATA_DIR / "weights_asset_set2.csv")

    print("\n── Experiment 2 Complete ──────────────────────────────")
    return outputs


if __name__ == "__main__":
    run_exp_2()
