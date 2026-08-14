# Changelog

## 0.19.0 - SVI arbitrage-free surface fit

- `options/svi.py`: the raw-SVI smile parameterisation, implemented from the
  published formula (Gatheral 2004; Gatheral and Jacquier 2014) rather than
  ported from any specific codebase. `svi_total_variance` evaluates it and
  `fit_svi_slice` calibrates the five parameters to one expiry by bounded least
  squares, enforcing the feasibility constraints (b >= 0, |rho| < 1, sigma > 0)
  directly and accepting per-quote weights so tighter markets pull harder.
- `is_butterfly_arbitrage_free` checks a fitted slice with Gatheral's g(k)
  function, whose sign is the sign of the risk-neutral density, using analytic
  SVI derivatives cross-checked against finite differences. This gives a
  parametric alternative to the interpolated surface, with the same refusal to
  assume arbitrage away rather than test for it.
- `fit_svi_points` / `fit_svi_surface` fit every expiry of a surface and report
  the parameters, the RMS fit error and the no-arbitrage verdict per expiry.
- 13 new tests (270 total).

## 0.18.0 - next-open fills

- Prices now carry an optional open: `PriceData.open` (and `has_open`), produced
  by the synthetic generator as yesterday's close plus a small overnight gap, and
  carried through the yfinance fetch, the parquet cache and alignment. Open is
  exposed only when every ticker in a panel has it, so a partial panel never
  silently disables the mode for some names.
- `backtest/engine.py` gains `portfolio_returns_next_open` and a `fill` argument
  on `run_backtest`. Under next-open execution the signal formed at close(T) is
  filled at open(T+1) and earns the open-to-open return, giving up the overnight
  gap it can no longer trade on. The held book, turnover and costs are unchanged,
  since they do not depend on which price P&L is marked at; the fill is causal by
  construction (the weight is still lagged a day).
- The fill mode threads through `walk_forward` and a new `--next-open` CLI flag,
  which degrades to close-fill with a clear message when the loaded panel has no
  open prices.
- The existing cached parquet files predate the open column, so real-data
  next-open runs need a cache refresh; the mode is fully exercised on synthetic
  data, where opens are always present.
- 10 new tests (257 total).

## 0.17.0 - meta-labeling the ML sleeve

- `signals/meta_labeling.py`: meta-labeling (Lopez de Prado). `primary_side`
  extracts the directional model's side, `meta_labels` marks whether that side
  was right, `train_meta_model` fits a secondary classifier to grade it, and
  `meta_sized_weights` sizes each bet by the meta-model's conviction above a coin
  flip, vetoing the ones below it. `meta_train_predict` wires the two models into
  one walk-forward callback and degrades to the primary's own sizing if the
  meta-model cannot be fit.
- The meta pooling uses the same strict `< fit_end` cut as the primary, so the
  secondary never trains on an out-of-sample label.
- `scripts/build_meta_results.py` compares the sleeve with and without the meta
  layer on the same data pull. Result: meta-labeling nudges the Sharpe up
  (-0.70 to -0.64) and, more usefully, cuts the drawdown (-24.4% to -20.3%) by
  vetoing some bad bets, but it does not turn a negative-edge signal positive. A
  filter on a losing signal is still a losing signal, and the README says so.
- 10 new tests (247 total).

## 0.16.0 - fat-tail risk: Cornish-Fisher, EVT, and stress scenarios

- `risk/metrics.py` gains `cornish_fisher_var`: the Gaussian VaR quantile
  corrected for sample skewness and excess kurtosis, so a left-skewed, fat-tailed
  book reports a larger loss than the normal assumption would.
- `evt_tail` fits a Generalized Pareto distribution to losses over a high
  threshold (peaks-over-threshold) and reads deep-quantile VaR and expected
  shortfall off that fit, plus the tail index `xi`. This extrapolates into the
  tail rather than being capped by the worst observed loss, which is where
  historical VaR is noisiest.
- `risk/stress.py`: `stress_test` estimates strategy P&L under market shocks. The
  default shocks are the benchmark's own worst day, week and month in the sample
  (computed, never hard-coded), passed through the strategy's beta; custom named
  scenarios can be supplied. The beta approximation is documented as a floor on
  the pain, since correlations rise in a crash.
- The text tearsheet now prints the Cornish-Fisher VaR and the EVT 99% VaR /
  expected shortfall / tail index alongside the historical and Gaussian figures,
  so the gap between them shows the fat tail directly.
- 11 new tests (237 total).

## 0.15.0 - GARCH(1,1) conditional-volatility forecasts

- `risk/garch.py`: `garch_forecast_vol` fits a GARCH(1,1) and returns the
  one-step-ahead conditional volatility. Unlike trailing realised vol, it reacts
  to a shock immediately and then decays, which is the volatility clustering
  realised vol lags. `arch` is a new optional `garch` extra, imported lazily so
  the rest of the package runs without it.
- `garch_vol_series` produces a causal forecast for every day of a return path,
  refitting on expanding history every 21 days and carrying the last forecast
  between refits (a documented cost/granularity trade-off). Every value is fit on
  data strictly before its day, proven by a causality test.
- `garch_vol_target` sizes a return stream toward a volatility target using the
  forecast. Because the forecast already looks one step ahead, no extra one-day
  lag is applied, unlike the realised-vol targeting in `portfolio` and `risk.sizing`.
- `regime/detector.py` gains `garch_regime`, a third causal regime definition
  alongside the vol-ratio and HMM detectors: defensive when the GARCH forecast
  runs hot versus its own trailing-median baseline.
- CI now installs the `garch` extra, and the GARCH tests `importorskip("arch")`
  so the suite still passes where the extra is absent.
- 11 new tests (226 total).

## 0.14.0 - one-file HTML tearsheet

- `performance/rolling.py`: `rolling_sharpe` and `rolling_beta` series, plus a
  `per_year_table` (each year's actual compound return, annualised vol and
  Sharpe, and worst drawdown). The per-year return is the real figure, not an
  annualised one, so a partial first or last year is reported as what it was.
- `performance/html_report.py`: `build_html_report` renders a single
  self-contained HTML tearsheet, headline metrics, the per-year table, an
  equity-and-drawdown chart and rolling Sharpe/beta. Every figure is an embedded
  base64 PNG, so the report is one file with no external assets and opens the
  same offline.
- `scripts/build_report.py` writes the tearsheet for the blended book against the
  market ETF as the beta benchmark, and refuses to build from synthetic data.
  The output is committed at `docs/results/report.html` and linked from the README.
- 10 new tests (215 total).

## 0.13.0 - ML model as a pipeline, with calibration and sample uniqueness

- `signals/ml_signal.py` gains `build_classifier`, one factory that returns the
  gradient-boosting model as an sklearn `Pipeline`, replacing the same estimator
  config that was duplicated at three fit sites. A regression test pins that the
  single-step pipeline predicts identically to the bare estimator, so the
  documented backtest numbers do not move under the refactor.
- Probability calibration: `build_classifier(calibrate="isotonic"|"sigmoid")`
  wraps the pipeline in a `CalibratedClassifierCV`. This matters because
  positions are sized by `P(up) - 0.5`, so a miscalibrated probability is a
  mis-sized bet, not just a wrong label.
- `signals/sample_weights.py`: label-uniqueness weights (Lopez de Prado). The
  pooled training set stacks every asset every day, so a market-wide move is
  counted many times; weighting each row by one over its label concurrency
  down-weights crowded days. For these single-bar (next-day) labels this is the
  exact single-bar case of average uniqueness, stated as such rather than dressed
  up as the overlapping triple-barrier version.
- `train_predict` now builds through the factory and routes both calibration and
  sample weights via `MLConfig.calibrate` and `MLConfig.uniqueness_weighting`.
  Both default off, so the traded signal and its published numbers are unchanged;
  measuring their effect on the ML sleeve's out-of-sample Sharpe is left as its
  own piece of work rather than folded in silently.
- 13 new tests (205 total).

## 0.12.0 - hierarchical risk parity, measured against inverse-variance

- `portfolio/hrp.py`: Hierarchical Risk Parity (Lopez de Prado, 2016).
  `correlation_distance` maps correlation to a proper distance; `cluster_order`
  quasi-diagonalises via single-linkage clustering on that distance;
  `recursive_bisection` splits the risk budget down the clustered order; and
  `hrp_weights` ties them together into long-only weights. `inverse_variance_weights`
  is the 1/var baseline it is measured against. No covariance matrix is inverted.
- `scripts/build_hrp_results.py`: a rolling monthly-rebalanced, causal,
  trailing-year-covariance contest on the 44-name universe, gross of costs.
  Result: HRP prints the best Sharpe (1.15 vs 0.97 for inverse-variance and 1.10
  for equal weight), the lowest realised vol (13.5%) and the shallowest drawdown,
  which is what the paper claims. The honest twist is concentration: HRP's
  effective N (1 / Herfindahl) is lower than inverse-variance's, so it is more
  concentrated by name yet realises less risk, because it diversifies in risk
  space rather than by owning a little of everything.
- 8 new tests (192 total).

## 0.11.0 - combine the three sleeves into one vol-targeted book

- `portfolio/allocator.py`: a new package for running the momentum, pairs and ML
  sleeves as one book. `inverse_vol_allocations` splits capital across sleeves
  inversely to trailing volatility (risk parity), lagged a day so the weight for
  a given day uses only vol known through the prior day; a sleeve still warming
  up gets zero and the rest are renormalised.
- `combine_weights` stitches per-sleeve weight matrices into one book on the
  union of tickers, as the allocation-weighted sum of each sleeve's weights.
- `blend_returns` blends the sleeve return streams by those allocations, and
  `volatility_target` scales the blended stream to a target annual vol using the
  same lagged, capped convention as `risk.sizing.vol_target_scale`.
- `scripts/build_blend_results.py` runs the real-data walk-forward for all three
  sleeves and combines them. The honest result: the vol target lands at 9.5%
  against a 10% aim, but equal-risk allocation gives the negative-edge ML sleeve
  the same risk budget as the positive pairs sleeve, so the blend prints a -0.33
  Sharpe. A naive equal-weight blend scores -0.30 over the same span, so risk
  parity is not the culprit; a losing sleeve is. Documented as such in the README.
- 12 new tests (184 total).

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
