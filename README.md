# QCA — Market State Classification & Dynamic Portfolio Optimization

A quantitative finance research project that classifies market regimes via K-means
clustering of rolling volatility and constructs a Markov-weighted dynamic portfolio
that blends four static strategies based on predicted market state transitions.

## Project Structure

```
.
├── data/                     # Downloaded price data (CSV exports, state series)
├── src/                      # Core implementation modules
│   ├── data_loader.py        # yfinance data download, alignment, returns
│   ├── market_states.py      # K-means market state classification (k=10)
│   ├── portfolio_static.py   # EW, MinVar, MaxDiv, ERC weight computation
│   ├── markov_transition.py  # Bayesian Markov transition matrix + Gibbs sampling
│   ├── dynamic_portfolio.py  # Dynamic portfolio construction & performance metrics
│   └── visualization.py      # All matplotlib/seaborn plots
├── exp/                      # Experiment runners
│   ├── exp_1_run.py          # Data prep and K-means state labelling
│   ├── exp_2_run.py          # Static portfolio weights
│   ├── exp_3_run.py          # Bayesian Markov transition matrix
│   ├── exp_4_run.py          # Dynamic portfolio + full performance evaluation
│   └── run_all.py            # End-to-end pipeline orchestrator
├── tests/                    # pytest test suite (115 tests)
│   ├── test_exp1_data_preparation.py
│   ├── test_exp2_static_portfolios.py
│   ├── test_exp3_markov_transition.py
│   └── test_exp4_dynamic_portfolio.py
├── results/                  # Output artefacts
│   ├── RESULTS.md            # Experiment findings summary
│   ├── fig*.png              # All figures (Figs 1-7 from the paper)
│   └── *.csv                 # Annual and full-period performance tables
└── README.md
```

## Experiments

| # | Title | Key Output |
|---|-------|-----------|
| 1 | Data Preparation & K-Means Market State Classification | 10 ordered volatility states per asset set |
| 2 | Static Portfolio Weight Computation | EW, MinVar, MaxDiv, ERC weights |
| 3 | Bayesian Markov Transition Matrix | 10×10 posterior mean transition matrix |
| 4 | Dynamic Portfolio Construction & Evaluation | Per-state strategy blending via Markov weights |

## Asset Sets

- **Asset Set 1**: AAPL, LLY, JPM, AMZN, GOOGL, UPS, PG, XOM, NEE, AMT, LIN  
  Period: 2005-06-20 → 2024-06-20 (~4,783 trading days)
- **Asset Set 2**: QQQ, SPY, BTC-USD, GLD, TLT  
  Period: 2015-01-06 → 2024-06-20 (~2,380 trading days)

## Methodology

1. **Rolling Volatility** — 22-day sample std of equal-weighted portfolio returns
2. **K-Means Clustering** — k=10, n_init=50, ordered by centroid magnitude
3. **Static Strategies** — SLSQP-optimised (long-only, fully-invested)
4. **Bayesian Markov Matrix** — Dirichlet(1) prior; posterior mean = (1+N)/(10+ΣN)
5. **Dynamic Portfolio** — `r_dyn = W[state_t] · [r_EW, r_MinVar, r_MaxDiv, r_ERC]`
6. **Custom Sharpe** — compound_return / std(daily_returns, ddof=1) (NOT annualised)

## Quick Start

```bash
# Install dependencies
pip install yfinance scikit-learn scipy numpy pandas matplotlib seaborn arviz

# Run all experiments
python exp/run_all.py

# Run tests
python -m pytest tests/ -v
```

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full metrics tables and figure descriptions.

### Key Findings (Asset Set 1, 2005-2024)

| Strategy | Compound Return | Daily Vol | Custom Sharpe |
|----------|----------------|-----------|---------------|
| EW | 2333% | 1.20% | 1947 |
| MinVar | 877% | 1.03% | 855 |
| MaxDiv | 2476% | 1.16% | 2141 |
| ERC | 1946% | 1.15% | 1694 |
| **Dynamic** | **2851%** | **1.11%** | **2579** |

- The Dynamic strategy achieves the highest Sharpe ratio and compound return by
  routing to MaxDiv in low-volatility states and MinVar/EW during high-volatility regimes.
- Asset Set 1 SLEM λ₂ ≈ 0.9563 (expected 0.9584); mixing time bound: [86, 145] days.
- Asset Set 2 SLEM λ₂ ≈ 0.9263 (expected 0.9277); mixing time bound: [49, 83] days.
- Gelman-Rubin max R-hat = 1.0002 << 1.1 (Gibbs chains converged trivially due to conjugate Dirichlet posterior).
