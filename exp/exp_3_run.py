"""
Experiment 3: Bayesian Markov Transition Matrix Construction.

Counts state transitions, applies uniform Dirichlet priors (alpha=1),
computes posterior mean via Gibbs sampling, validates with Gelman-Rubin
diagnostics, and computes SLEM and mixing time bounds.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.markov_transition import (
    build_transition_matrix,
    compute_posterior_mean,
    count_transitions,
)
from src.visualization import plot_transition_matrix_heatmap

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Expected reference values from the paper
EXPECTED_SLEM_SET1 = 0.9584
EXPECTED_SLEM_SET2 = 0.9277
EXPECTED_MIXING_SET1 = 109
EXPECTED_MIXING_SET2 = 62
EXPECTED_P1_ROW = [0.901227, 0.082535, 0.004582, 0.002922, 0.001539,
                   0.001453, 0.001407, 0.001503, 0.001421, 0.001412]


def run_exp_3(exp1_outputs: dict = None) -> dict:
    """
    Execute Experiment 3: Bayesian Markov transition matrix construction.

    Parameters
    ----------
    exp1_outputs : dict, optional
        Outputs from Experiment 1. If None, loads from disk.

    Returns
    -------
    dict with transition matrices and diagnostics for both asset sets.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 3: Bayesian Markov Transition Matrix")
    print("=" * 60)

    if exp1_outputs is None:
        with open(DATA_DIR / "exp1_outputs.pkl", "rb") as f:
            exp1_outputs = pickle.load(f)

    state_series1 = exp1_outputs["state_series1"]
    state_series2 = exp1_outputs["state_series2"]

    # ── Asset Set 1 ───────────────────────────────────────────────────────────
    print("\n── Asset Set 1 ─────────────────────────────────────────")
    print(f"  State sequence length: {len(state_series1)}")

    P_mean1, N1, rhat1, diag1 = build_transition_matrix(
        state_series1,
        n_states=10,
        alpha=1.0,
        n_chains=4,
        n_iter=10000,
        burn_in=2000,
    )

    print("\n  Posterior mean P[0, :] (state 1 row):")
    print(f"  {P_mean1[0, :].round(6).tolist()}")

    # Validate against paper reference
    p1_ref = np.array(EXPECTED_P1_ROW)
    max_diff_p1 = np.max(np.abs(P_mean1[0, :] - p1_ref))
    print(f"  Max |P[0,:] - reference|: {max_diff_p1:.4f} (tolerance: 0.02)")

    print(f"\n  SLEM (lambda_2): {diag1['slem']:.4f} (expected ~{EXPECTED_SLEM_SET1})")
    print(f"  t_rel: {diag1['t_rel']:.2f}")
    print(f"  Mixing time bounds: [{diag1['t_mix_lower']:.1f}, {diag1['t_mix_upper']:.1f}]")
    print(f"  Expected mixing time ~{EXPECTED_MIXING_SET1} "
          f"{'✓' if diag1['t_mix_lower'] <= EXPECTED_MIXING_SET1 <= diag1['t_mix_upper'] else '?'}")

    # ── Asset Set 2 ───────────────────────────────────────────────────────────
    print("\n── Asset Set 2 ─────────────────────────────────────────")
    print(f"  State sequence length: {len(state_series2)}")

    P_mean2, N2, rhat2, diag2 = build_transition_matrix(
        state_series2,
        n_states=10,
        alpha=1.0,
        n_chains=4,
        n_iter=10000,
        burn_in=2000,
    )

    print(f"\n  SLEM (lambda_2): {diag2['slem']:.4f} (expected ~{EXPECTED_SLEM_SET2})")
    print(f"  t_rel: {diag2['t_rel']:.2f}")
    print(f"  Mixing time bounds: [{diag2['t_mix_lower']:.1f}, {diag2['t_mix_upper']:.1f}]")
    print(f"  Expected mixing time ~{EXPECTED_MIXING_SET2} "
          f"{'✓' if diag2['t_mix_lower'] <= EXPECTED_MIXING_SET2 <= diag2['t_mix_upper'] else '?'}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n── Generating Visualizations ──────────────────────────")
    plot_transition_matrix_heatmap(
        P_mean1,
        title="Asset Set 1: Bayesian Markov Transition Matrix",
        filename="fig3_asset_set1_transition_matrix.png",
    )
    plot_transition_matrix_heatmap(
        P_mean2,
        title="Asset Set 2: Bayesian Markov Transition Matrix",
        filename="fig3b_asset_set2_transition_matrix.png",
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n── Saving Outputs ─────────────────────────────────────")
    outputs = {
        "P_mean1": P_mean1,
        "N1": N1,
        "rhat1": rhat1,
        "diagnostics1": diag1,
        "P_mean2": P_mean2,
        "N2": N2,
        "rhat2": rhat2,
        "diagnostics2": diag2,
    }

    with open(DATA_DIR / "exp3_outputs.pkl", "wb") as f:
        pickle.dump(outputs, f)
    print(f"  Saved: {DATA_DIR / 'exp3_outputs.pkl'}")

    # Save transition matrices as CSV
    pd.DataFrame(P_mean1).to_csv(DATA_DIR / "transition_matrix1.csv")
    pd.DataFrame(P_mean2).to_csv(DATA_DIR / "transition_matrix2.csv")

    print("\n── Experiment 3 Complete ──────────────────────────────")
    return outputs


if __name__ == "__main__":
    run_exp_3()
