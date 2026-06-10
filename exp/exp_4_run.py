"""
Experiment 4: Dynamic Portfolio Construction and Full Performance Evaluation.

Identifies best static strategy per state, builds total return weight matrix,
constructs dynamic portfolio, and evaluates all strategies on annual and
full-period metrics with the paper's non-standard Sharpe ratio definition.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dynamic_portfolio import run_full_evaluation
from src.visualization import (
    plot_annual_sharpe,
    plot_portfolio_value,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_exp_4(
    exp1_outputs: dict = None,
    exp2_outputs: dict = None,
    exp3_outputs: dict = None,
) -> dict:
    """
    Execute Experiment 4: Dynamic portfolio construction and evaluation.

    Parameters
    ----------
    exp1_outputs, exp2_outputs, exp3_outputs : dict, optional
        Outputs from previous experiments. If None, loads from disk.

    Returns
    -------
    dict with all results.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 4: Dynamic Portfolio Construction")
    print("=" * 60)

    # Load outputs from previous experiments
    if exp1_outputs is None:
        with open(DATA_DIR / "exp1_outputs.pkl", "rb") as f:
            exp1_outputs = pickle.load(f)

    if exp2_outputs is None:
        with open(DATA_DIR / "exp2_outputs.pkl", "rb") as f:
            exp2_outputs = pickle.load(f)

    if exp3_outputs is None:
        with open(DATA_DIR / "exp3_outputs.pkl", "rb") as f:
            exp3_outputs = pickle.load(f)

    # ── Asset Set 1 ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("ASSET SET 1")
    print("═" * 60)

    results1 = run_full_evaluation(
        state_series=exp1_outputs["state_series1"],
        strategy_returns=exp2_outputs["strategy_rets1"],
        P_mean=exp3_outputs["P_mean1"],
        n_states=10,
    )

    # Validate binary vectors
    print("\n── Validating Binary Assignment Vectors (Asset Set 1) ──")
    p_vectors_1 = results1["p_vectors"]
    expected_vectors_1 = {
        "ERC":    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "MinVar": [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        "MaxDiv": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "EW":     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    }

    for strat, exp_vec in expected_vectors_1.items():
        if strat in p_vectors_1:
            got_vec = p_vectors_1[strat].astype(int).tolist()
            match = got_vec == exp_vec
            print(f"  p_{strat}: got={got_vec}, expected={exp_vec} {'✓' if match else '≠'}")

    # Annual metrics tables
    annual_sharpe1 = results1["annual_metrics"]["sharpe"]
    annual_ret1 = results1["annual_metrics"]["compound_return"]
    annual_vol1 = results1["annual_metrics"]["volatility"]

    print("\n── Annual Compound Returns (Asset Set 1) ───────────────")
    print(annual_ret1.round(4).to_string())

    print("\n── Annual Volatility (Asset Set 1) ─────────────────────")
    print(annual_vol1.round(4).to_string())

    print("\n── Annual Sharpe Ratios (Asset Set 1) ──────────────────")
    print(annual_sharpe1.round(2).to_string())

    # ── Visualizations (Asset Set 1) ─────────────────────────────────────────
    print("\n── Generating Visualizations (Asset Set 1) ────────────")
    plot_portfolio_value(
        results1["portfolio_values"],
        title="Asset Set 1: Portfolio Growth ($1 Initial Investment)",
        filename="fig4_asset_set1_portfolio_value.png",
    )
    plot_annual_sharpe(
        annual_sharpe1,
        title="Asset Set 1: Annual Sharpe Ratio by Strategy",
        filename="fig5_asset_set1_annual_sharpe.png",
    )

    # ── Asset Set 2 ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("ASSET SET 2")
    print("═" * 60)

    results2 = run_full_evaluation(
        state_series=exp1_outputs["state_series2"],
        strategy_returns=exp2_outputs["strategy_rets2"],
        P_mean=exp3_outputs["P_mean2"],
        n_states=10,
    )

    # Validate Asset Set 2: EW should never be best
    p_ew2 = results2["p_vectors"].get("EW", np.zeros(10))
    print(f"\n── Asset Set 2: p_EW = {p_ew2.astype(int).tolist()} (expect all zeros)")

    annual_sharpe2 = results2["annual_metrics"]["sharpe"]
    annual_ret2 = results2["annual_metrics"]["compound_return"]
    annual_vol2 = results2["annual_metrics"]["volatility"]

    print("\n── Annual Compound Returns (Asset Set 2) ───────────────")
    print(annual_ret2.round(4).to_string())

    print("\n── Annual Sharpe Ratios (Asset Set 2) ──────────────────")
    print(annual_sharpe2.round(2).to_string())

    # ── Visualizations (Asset Set 2) ─────────────────────────────────────────
    print("\n── Generating Visualizations (Asset Set 2) ────────────")
    plot_portfolio_value(
        results2["portfolio_values"],
        title="Asset Set 2: Portfolio Growth ($1 Initial Investment)",
        filename="fig6_asset_set2_portfolio_value.png",
    )
    plot_annual_sharpe(
        annual_sharpe2,
        title="Asset Set 2: Annual Sharpe Ratio by Strategy",
        filename="fig7_asset_set2_annual_sharpe.png",
    )

    # ── Full-period summary ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("FULL-PERIOD PERFORMANCE SUMMARY")
    print("═" * 60)

    print("\n── Asset Set 1 ─────────────────────────────────────────")
    print(results1["full_metrics"].to_string(float_format="{:.6f}".format))

    print("\n── Asset Set 2 ─────────────────────────────────────────")
    print(results2["full_metrics"].to_string(float_format="{:.6f}".format))

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n── Saving Outputs ─────────────────────────────────────")
    outputs = {
        "results1": results1,
        "results2": results2,
    }

    with open(DATA_DIR / "exp4_outputs.pkl", "wb") as f:
        pickle.dump(outputs, f)
    print(f"  Saved: {DATA_DIR / 'exp4_outputs.pkl'}")

    # Save performance tables to CSV
    results1["full_metrics"].to_csv(RESULTS_DIR / "full_metrics_asset_set1.csv")
    results2["full_metrics"].to_csv(RESULTS_DIR / "full_metrics_asset_set2.csv")
    annual_sharpe1.to_csv(RESULTS_DIR / "annual_sharpe_asset_set1.csv")
    annual_sharpe2.to_csv(RESULTS_DIR / "annual_sharpe_asset_set2.csv")
    annual_ret1.to_csv(RESULTS_DIR / "annual_return_asset_set1.csv")
    annual_ret2.to_csv(RESULTS_DIR / "annual_return_asset_set2.csv")

    print("\n── Experiment 4 Complete ──────────────────────────────")
    return outputs


if __name__ == "__main__":
    run_exp_4()
