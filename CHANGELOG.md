# Changelog

## 0.6.0 - options leg, phase 1 (pricing and Greeks)

- New `quant_system/options/` package:
  - `pricing.py`: Black-Scholes for Europeans, CRR binomial tree for American
    early exercise. Degenerate corners (expiry, zero vol) return discounted
    forward intrinsic instead of dividing by zero. The tree raises rather than
    returning a plausible wrong number when the risk-neutral probability falls
    outside (0,1).
  - `greeks.py`: analytic Black-Scholes Greeks plus a generic central
    finite-difference engine, used for American options where no closed form
    exists. Units are stated explicitly (vega per 1.00 vol, theta per year)
    with converters to the per-point and per-day quotes desks use.
  - `implied_vol.py`: Brent solver with static no-arbitrage bound checks and
    documented reason codes on failure. Deep ITM options are solved through
    their OTM twin via put-call parity, where the price is all time value and
    the problem is far better conditioned.
- `OptionsConfig` added to `config.py`: rates, tree steps, solver bracket and
  finite-difference bump sizes, so none of these are hard-coded.
- 62 new tests (117 total). The pricing was cross-checked against numerical
  quadrature of the risk-neutral expectation, agreeing to 3e-10.
- `scripts/verify.sh` prints the test count and line coverage in one command,
  so any figure quoted in the README or on a CV is reproducible.

## 0.5.0

- Pair selection is now causal: each walk-forward fold re-runs the
  Engle-Granger scan (with the FDR correction) on data available at the fold
  boundary and trades the pair it would actually have chosen at the time. A
  fold where no candidate survives stays flat. Previously the pair was picked
  once on the full sample, which let the selection peek at the test years.
  On real data 2016-2024 this fix reduced the pairs OOS Sharpe from 0.61 to
  0.42 - the results got worse, which is the point: the difference was
  look-ahead bias, not alpha.
- Added a results section to the README: out-of-sample walk-forward table and
  equity chart built from real yfinance data by `scripts/build_results.py`.
  The script refuses to run on the synthetic fallback so published numbers
  can never come from generated data.

## 0.4.0

- Added an execution model (`backtest/execution.py`, CLI flag
  `--max-participation`). Daily trading in a name is capped at a fraction of
  its average daily volume; whatever doesn't fill carries to later days, so
  the held book chases the target instead of teleporting to it. The gap
  between target and held is reported per day.
- Fixed `cost=None`, which the docstring promised meant zero costs but which
  actually substituted the default cost config. Frictionless baselines (the
  capacity sweep among them) were quietly paying spread and impact.

## 0.3.0

- Added purged, embargoed K-fold cross-validation for the ML classifier
  (`signals/cv.py`, CLI flag `--cv`). Folds are contiguous time blocks;
  training rows whose label interval overlaps the test window are purged, and
  a configurable embargo drops the days right after each test block. Reports
  per-fold accuracy and AUC.

## 0.2.0

- Fixed a one-day label leak in the ML training window: the row dated at the
  in-sample boundary was labelled with the first out-of-sample day's return.
  Rows with an unknown next-day return also got a fake "down" label instead of
  being dropped. A regression test now proves training is identical whether or
  not post-boundary data exists in the panel.
- Pair selection is now corrected for multiple testing. The scan reports
  Benjamini-Hochberg q-values next to raw p-values, and the chosen pair has to
  survive the correction at the configured false discovery rate.
- Assorted docstring cleanup.

## 0.1.0

First working version. A backtesting framework for a few systematic equity
strategies on daily data, with realistic costs and out-of-sample testing.

### Strategies
- Cross-sectional momentum (12-1, monthly rebalance).
- Engle-Granger cointegration pairs trade, with a kill-switch that stops trading a
  pair once the relationship breaks down.
- Gradient-boosted next-day direction model, sized by predicted probability, with
  SHAP feature importances.

### Engine and evaluation
- One-day signal lag so a position can't use a price it wouldn't have had yet.
- Trading costs as a half-spread plus square-root market impact.
- Walk-forward out-of-sample validation.
- Performance stats (Sharpe, Sortino, Calmar, drawdown, hit rate, turnover, VaR,
  CVaR) and a Fama-French 3-factor regression with Newey-West standard errors.
- Probabilistic and deflated Sharpe ratios, bootstrap confidence intervals, and a
  capacity estimate for how much capital the strategy can hold before costs win.

### Tools
- Multi-ticker SEC EDGAR filing monitor with text or phone-call alerts (Twilio).
- A market-snapshot analyst that returns structured JSON through the Claude API.
