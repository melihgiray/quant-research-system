# Changelog

## 0.10.0 - false-discovery-rate control on the ML feature search

- `signals/ml_signal.py` gains `permutation_importance_pvalues`: a one-sided
  p-value per feature against a permutation null (shuffle the label, no real
  feature-label relationship remains), with the finite-sample +1 correction so a
  p-value is never exactly zero.
- `signals/feature_selection.py`: a Benjamini-Hochberg pass over those p-values,
  the same correction already used for the pairs scan, reporting a q-value and a
  keep flag per feature.
- `signals/ml_signal.py` gains `ml_feature_significance`, the end-to-end
  diagnostic: build the pooled training set, split off a held-out portion, fit
  the model, score permutation-null p-values on the held-out data, and FDR-correct
  across the eight features. New `--feature-fdr` CLI flag prints it.
- The point of the correction, on synthetic data: only `zscore_21` survives
  (q=0.04), while `mom_5` has a raw p=0.035 that looks significant on its own but
  is demoted to q=0.14 and dropped once the eight-way test is accounted for.
- 11 new tests (172 total).

## 0.9.0 - options leg, real chain data (one crisis day)

- `options/history.py`: a loader for the OptionsDX end-of-day CSV format, which
  is the free way to get a real historical option chain. It parses the wide
  call-and-put-per-row layout into the repo's long quote schema, so a historical
  snapshot feeds the surface builder exactly like a live chain. Empty vendor
  fields become NaN, the absent open-interest column is left NaN rather than
  faked, time-to-expiry comes from the fractional vendor DTE and is cross-checked
  against the calendar difference, and the shared hygiene filters report their
  drops the same way as live chains. Vendor Greeks and IVs are carried through as
  validation material, not truth.
- `scripts/validate_optionsdx.py`: checks our Brent implied-vol solver against
  the vendor's IVs on a real SPY chain (2020-03-06, the COVID crash). Near the
  money the two agree to about 2 vol points, stable at 1.73-1.78 points across
  all 31 intraday snapshots; the wings diverge as expected from unknown vendor
  rate/dividend assumptions and American-vs-European style. This is the
  options-side analogue of the Phase 1 quadrature cross-check: independent
  implementation agreement as evidence the solver is right on real quotes.
- The same script draws `docs/results/vol_surface_spy_crisis.png`: the crisis
  smile, the inverted ATM term structure (53% at 7d down to 22% at two years,
  the opposite of the calm 2026 live surface), and intraday 30-day ATM vol
  moving opposite spot.
- Honesty notes kept explicit: one day is not enough to backtest, so strategy
  backtests still use the synthetic chain; the OptionsDX file is not committed
  (terms not confirmed for redistribution) and is read from a caller-supplied
  path, with a hand-built schema-clone fixture under `tests/fixtures/` so the
  suite needs no download.
- 12 new tests (161 total), 11 offline against the fixture plus one real-file
  regression guard that skips when the sample is absent.

## 0.8.0 - options leg, phase 3 (strategy backtests)

- `options/book.py`: contracts, portfolio state and fill rules. Buys take the
  ask and sells hit the bid, never the mid, which matters far more for options
  than equities: single-stock options are routinely 2-5% wide, so a mid-filled
  covered-call backtest overstates the premium it claims to collect by most of
  that premium. Marks stay at mid so the spread is not double-counted.
  Expiry settles physically, so an assigned short call delivers shares and an
  assigned short put takes delivery.
- `options/provider.py`: a chain of quotes through time, generated because
  yfinance has no options history. Implied vol is trailing realised vol (lagged
  a day, so the surface never peeks) times `(1 + vol_premium)`. That premium is
  an explicit input, documented as an assumption rather than a finding.
- `options/strategies.py`: covered call, cash-secured put, delta-hedged short
  straddle, plus a buy-and-hold benchmark. Strategies read position state from
  the book rather than tracking their own, so assignment cannot desynchronise
  them.
- `options/backtest.py`: event-driven engine holding the equity side's timing
  rule. Orders decided at the close of T fill against T+1's quotes, enforced
  structurally by a pending-order queue rather than by convention. Records
  equity, daily Greeks, exposures, fills and assignment events.
- Fixed a strategy bug the delta plot exposed: on a straddle roll the share
  hedge was left on the books after every option leg was closed, leaving an
  outright stock position. Unwinding it cut mean absolute delta from 17.4 to
  11.4 shares and peak from about 150 to 86.
- `scripts/build_options_results.py` reproduces the table, the premium sweep
  and the plot.
- 15 new tests (149 total). The look-ahead guard for the options path measures
  the lag rather than asserting it: a strategy peeking one day ahead scores
  Sharpe 0.11 because it trades after the move it foresaw, while the same
  strategy peeking two days ahead scores 15.27. The gap is the one-day lag.

## 0.7.0 - options leg, phase 2 (vol surface)

- `options/chain.py`: option chain ingestion from yfinance, plus a labelled
  SYNTHETIC generator with a realistic skew and term structure. Hygiene filters
  drop untradeable quotes (zero bid, crossed, spread too wide, expired) and
  report the count per reason instead of silently shrinking the data. On a live
  SPY chain that removed 399 of 2,310 raw quotes, 219 of them zero-bid.
  Expiries are sampled across the term structure by default, since taking the
  nearest few on a daily-expiry underlying gives no term structure at all.
- `options/surface.py`: surface in log-forward-moneyness and expiry,
  interpolated linearly in total variance and clamped rather than extrapolated
  at the wings. Total variance is the right coordinate because the
  no-calendar-arbitrage condition is exactly its monotonicity in maturity;
  interpolating in implied vol instead can create arbitrage from clean inputs.
- `options/arbitrage.py`: butterfly (convexity in strike), strike monotonicity
  and calendar (total variance monotone in expiry) checks. Violations are
  reported with location and magnitude, never smoothed away, and each is graded
  against the local bid-ask spread: a violation smaller than the spread you
  would have to cross is a mid-price artifact, not an opportunity. On live SPY,
  330 butterfly violations were flagged and only 61 exceeded the local spread,
  clustered in the 878-day LEAPS where quotes go stale.
- Chain hygiene and surface tolerances added to `OptionsConfig`.
- `scripts/build_vol_surface.py` reproduces the surface and plot. In live mode
  it exits non-zero rather than falling back to synthetic data.
- 17 new tests (134 total). The arbitrage tests run in both directions: a clean
  synthetic surface must report zero violations, and injected butterfly and
  calendar violations must each be caught.

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
