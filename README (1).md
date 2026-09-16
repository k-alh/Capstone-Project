# Time Series Forecasting Capstone — Retail Demand (Riyadh / Grocery)

**Training programme:** SDAIA Academy — *Time Series Forecasting for AI Systems*
**Cohort dates:** Start date: 13/9/2026 End date: 15/9/2026
**Authors:** Naif Alsmari , khalid alhaidary , Ziyad Saleh Alhumayd , Abdulmajeed Alnashwan , and Abdulaziz Alfuraih

**SDAIA Academy on GitHub:** 

## Project idea

This is the capstone for the SDAIA Academy *Time Series Forecasting for AI Systems* course: take
one forecasting problem end to end — structure diagnostics, classical statistical models, an
ML/GBM model, a real walk-forward backtest, scale-free accuracy metrics, and calibrated
probabilistic intervals — on a single series, then write up which model family I'd actually deploy
and why, reasoning from more than raw accuracy.

I went beyond the minimum "at least one" version of each rubric section deliberately: **both**
SARIMAX and Holt-Winters/ETS (not just one classical model), **both** expanding and rolling
backtest windows (not just one), and **all four** interval methods the rubric names separately —
Prophet, quantile LightGBM, sktime, and split conformal — rather than picking a single one.

## Dataset

`data/retail_demand.csv`, filtered to the **Riyadh / Grocery** region/category pair: 1,096 daily
observations (2023-01-01 to 2025-12-31), strong weekly seasonality, a mild multi-year upward
trend, Hijri-style drifting holiday bumps, sporadic promo shocks, and no zero or near-zero days.
I chose it over the other three course datasets because:

- it's the same series the course's own Day 1–2 labs use as their running example, so my
  diagnostics and fits can be sanity-checked against the course's own worked numbers;
- it has enough history and clean-enough seasonality to properly exercise every required
  section (unlike the 108-point economic-indicator series or the structural-break workforce
  series); and
- it has no sparsity problem, which makes MASE (rather than WAPE) the natural primary
  scale-free metric — a genuine methodological choice the notebook argues for explicitly,
  rather than defaulting to whichever metric the brief mentions first.

Every number in this notebook comes from that synthetic, seeded dataset
(`data/generate_series.py`, seed `20260912`) — none of it describes a real retailer.

## How to open and run this notebook

**Recommended — Google Colab.** Click the "Open in Colab" badge at the top of `capstone.ipynb`
(update the badge's `YOUR_GITHUB_USERNAME/YOUR_REPO_NAME` placeholder to this repo's actual path
once pushed). The notebook's first code cell installs everything it needs
(`pandas`, `numpy`, `matplotlib`, `statsmodels`, `scikit-learn`, `lightgbm`, `prophet`, `sktime`)
and downloads `common/metrics.py`, `common/backtest.py`, and `data/retail_demand.csv` from the
course repository automatically if they aren't already present locally — no local setup, no API
key, no account needed beyond Colab itself. Just open it and run all cells top to bottom.

**Locally**, if preferred:

```bash
git clone <this-repo-url>
cd <this-repo>
pip install pandas numpy matplotlib statsmodels scikit-learn lightgbm prophet sktime jupyter
jupyter notebook capstone.ipynb
```

`common/` and `data/` are included in this repo so the notebook's `fetch()` helper finds them
locally without hitting the network at all when run from a clone.

Note: Prophet's very first `.fit()` call in a fresh environment compiles and caches a `cmdstan`
model in the background, which can take anywhere from a few seconds to a couple of minutes —
that pause is normal, not a hang.

## What the notebook contains

| # | Section | Highlights |
|---|---|---|
| 1 | Structure & diagnostics | STL decomposition; ACF/PACF on raw and differenced series; ADF test (raw series p=0.0023, differenced p≈4.6e-17), with the differencing decision tied to the ACF's seasonal spike rather than the ADF result alone, since ADF says nothing about seasonality |
| 2 | Classical models | A 216-candidate SARIMAX grid search selected by AIC/BIC (winner: `(1,1,2)x(0,1,1,7)`), plus an 8-candidate Holt-Winters/ETS grid also selected by AIC — which picked a config that turned out to have a degenerate fit (`beta=0, gamma=0`) and lost to seasonal-naive on this specific holdout, an honest result the notebook explains rather than hides. Ljung-Box residual diagnostics on both models |
| 3 | ML/GBM | LightGBM with lag/rolling/calendar features; an explicit walkthrough of the shift-based leakage trap; a leakage-safe recursive multi-step forecast with a lineage log showing how many lag features are backed by real history vs. the model's own earlier predictions |
| 4 | Backtesting | `common/backtest.py`'s walk-forward harness; **6 folds under both expanding and rolling windows** (not just one), 4 models including a seasonal-naive baseline, fold-boundary leakage discussed concretely |
| 5 | Metrics | MAE/RMSE plus **both** WAPE and MASE via `common/metrics.py`, with the choice of MASE as primary justified against this series' non-sparse shape |
| 6 | Probabilistic forecasting | **Prophet**'s native intervals, **quantile LightGBM** scored with pinball loss, **sktime**'s `predict_interval` API, and a **split-conformal** interval — all four scored on coverage *and* width together, with the conformal method's genuine under-coverage (40% vs. an 80% target) explained rather than hidden |
| 7 | Model comparison | A written recommendation reasoning from history length, interpretability, interval support, and compute budget — referencing the concrete numbers produced in Sections 2–6, naming all four tool families (statsmodels, Prophet, sktime, LightGBM) |

## Repository structure
