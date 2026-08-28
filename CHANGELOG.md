# Changelog

## 0.60.0 - downside beta

- `performance/active.py` gains `downside_beta`: beta measured only on days the
  benchmark is below its mean, isolating how much a strategy participates in market
  declines. A defensive book has a downside beta below its overall beta.
- 2 new tests.

## 0.59.0 - Treynor ratio

- `performance/active.py` gains `treynor_ratio`: annualised excess return per unit
  of market beta, the systematic-risk analogue of the Sharpe ratio.
- 3 new tests.

## 0.58.0 - CAPM alpha and beta

- `performance/active.py` gains `alpha_beta`: annualised alpha and beta from a
  regression of returns on a benchmark, the return left over once market exposure
  is priced out.
- 3 new tests.

## 0.57.0 - half-life of mean reversion

- `stats.py` gains `half_life`: the Ornstein-Uhlenbeck half-life of a spread,
  estimated by regressing the change on the lagged level. It recovers the theory
  on simulated OU processes (faster reversion, shorter half-life) and returns a
  very large value for a random walk. This is the quantity a pairs trade uses to
  size a holding period.
- 3 new tests.

## 0.56.0 - return diagnostics in the tearsheet (integration)

- `performance/tearsheet.py` now prints a diagnostics block, ulcer index, 95%
  conditional drawdown at risk, Omega, tail ratio, and the Hurst exponent of the
  equity curve, wiring the risk and time-series primitives (0.35, 0.39, 0.40, 0.45,
  0.46) into the text tearsheet the CLI produces.
- 1 new test.

## 0.55.0 - ERC sleeve allocator and comparison (integration)

- `portfolio/allocator.py` gains `erc_allocations`, a causal rolling equal-risk-
  contribution allocation across sleeves, alongside inverse-vol and HRP.
- `scripts/build_hrp_blend_results.py` now includes an ERC blend, wiring the ERC
  allocator (0.33) into the measured sleeve-allocation comparison.
- The honest result: ERC is the worst of the risk-based blends (Sharpe -0.13), below
  even equal weight, and not because of a bug (its risk contributions are equal to
  three decimals). Equalising risk under correlation holds pairs at 74% where
  inverse-variance holds 92%, moving the freed capital into the higher-vol losing
  sleeves; balancing risk is not balancing skill. Written up in the README.
- 1 new test.

## 0.54.0 - benchmark metrics in the tearsheet (integration)

- `performance/html_report.py` now renders a "Versus benchmark" block, the
  information ratio, tracking error, and up/down capture ratios (from 0.48 and
  0.49), whenever a benchmark is supplied. `scripts/build_report.py` already passes
  the market ETF, so the committed blended-book tearsheet now shows the book
  against simply holding the market.
- 1 new test; the committed `docs/results/report.html` is regenerated.

## 0.53.0 - shrinkage in the allocator comparison (integration)

- `scripts/build_hrp_results.py` now runs Ledoit-Wolf-shrunk variants of HRP and
  inverse-variance alongside the originals, wiring the covariance-shrinkage
  primitive (0.41) into an actual measured backtest.
- The honest result: shrinkage barely moves either allocator (HRP 1.15 to 1.12,
  inverse-variance unchanged), because neither inverts the covariance, which is the
  problem shrinkage exists to fix. Documented in the README as a principled
  negative result rather than dropped.
- No new tests (uses existing, tested code); the README HRP table is refreshed.

## 0.52.0 - SADF explosiveness test

- `stats.py` gains `sadf`: the supremum Augmented Dickey-Fuller statistic
  (Phillips-Shi-Yu; Lopez de Prado, ch. 17) for detecting explosive, bubble-like
  behaviour. For each end point it takes the largest Dickey-Fuller statistic over
  backward-expanding windows; a random walk stays below ~1, while an explosive AR
  series pushes it sharply positive.
- 3 new tests.

## 0.51.0 - combinatorial purged cross-validation

- `signals/combinatorial_cv.py`: `combinatorial_purged_splits` partitions the data
  into N contiguous groups and tests every choice of k of them (Lopez de Prado, ch.
  12), yielding C(N, k) train/test splits with an embargo dropping neighbouring
  training samples around each test group, so many backtest paths come from one
  dataset without boundary leakage.
- 5 new tests.

## 0.50.0 - rolling Sortino

- `performance/rolling.py` gains `rolling_sortino`: a trailing annualised Sortino
  (excess return over downside deviation), the downside-only companion to the
  existing rolling Sharpe.
- 3 new tests.

## 0.49.0 - capture ratios

- `performance/active.py` gains `capture_ratios`: the up-capture and down-capture
  against a benchmark (mean strategy return over mean benchmark return on the
  benchmark's up and down days). A defensive strategy shows high up capture and low
  down capture.
- 3 new tests.

## 0.48.0 - information ratio and tracking error

- `performance/active.py`: benchmark-relative metrics. `tracking_error` is the
  annualised volatility of the return difference to a benchmark, and
  `information_ratio` is the annualised active return over that tracking error, a
  measure of skill relative to the benchmark rather than in absolute terms.
- 3 new tests.

## 0.47.0 - growth-optimal Kelly fraction

- `risk/sizing.py` gains `growth_optimal_fraction`: the single-stream Kelly leverage
  that maximises long-run log growth, mean over variance. It rises with edge and
  falls with variance, complementing the existing portfolio `kelly_weights`.
- 3 new tests.

## 0.46.0 - tail ratio

- `risk/metrics.py` gains `tail_ratio`: the size of the right return tail over the
  left (e.g. 95th percentile over the absolute 5th), a quick read on asymmetry that
  is ~1 for symmetric returns, above 1 for right-skewed and below for left-skewed.
- 3 new tests.

## 0.45.0 - conditional drawdown at risk

- `risk/metrics.py` gains `conditional_drawdown_at_risk`: the average of the worst
  tail of the underwater curve, a drawdown analogue of expected shortfall that is
  less hostage to a single worst point than max drawdown.
- 3 new tests.

## 0.44.0 - effective number of bets

- `risk/covariance.py` gains `effective_number_of_bets`: the exponentiated entropy
  of how variance spreads across a covariance's principal portfolios. It equals N
  for N uncorrelated assets and falls toward 1 as they collapse onto one common
  factor, a diversification measure that sees through correlation.
- 3 new tests.

## 0.43.0 - sample autocorrelation

- `stats.py` gains `autocorrelation`: the sample autocorrelation at a lag. Near 0
  for white noise; for an AR(1) with coefficient phi it recovers ~phi at lag 1 and
  ~phi**2 at lag 2, as the tests check.
- 3 new tests.

## 0.42.0 - return-sign entropy

- `entropy.py`: `returns_to_bits` encodes returns as up/down symbols and
  `plugin_entropy` estimates the Shannon entropy per symbol from the empirical word
  distribution (Lopez de Prado, ch. 18), a model-free measure of how predictable
  the sequence is. Fair random bits read ~1 bit/symbol; structure shows up as lower
  entropy at longer words.
- 5 new tests.

## 0.41.0 - Ledoit-Wolf covariance shrinkage

- `risk/covariance.py`: `ledoit_wolf_shrinkage` shrinks the sample covariance
  toward a scaled identity by the optimal intensity (Ledoit and Wolf 2004), which
  conditions the matrix for the allocators. The intensity falls as data grows, and
  the shrunk matrix is better conditioned than the raw sample covariance when
  observations are scarce.
- 4 new tests.

## 0.40.0 - Omega ratio

- `risk/metrics.py` gains `omega_ratio`: total gains above a threshold over total
  shortfalls below it, which uses the whole return distribution (skew and tails
  included) rather than just mean and variance.
- 4 new tests.

## 0.39.0 - ulcer index and pain ratio

- `risk/metrics.py` gains `ulcer_index` (root-mean-square drawdown, which unlike
  max drawdown penalises how long as well as how deep a series stays underwater)
  and `pain_ratio` (annualised return over the ulcer index).
- 5 new tests.

## 0.38.0 - Amihud illiquidity

- `microstructure.py` gains `amihud_illiquidity`, the average absolute return per
  dollar of volume (Amihud 2002); higher means thinner liquidity. It scales
  inversely with volume and skips zero-volume days.
- 2 new tests.

## 0.37.0 - Roll's implied spread

- `microstructure.py`: `roll_spread` recovers an effective bid-ask spread from the
  serial covariance of daily price changes (Roll 1984), needing only closes. A
  test plants a known bounce spread and recovers it to a few percent, and a
  bounce-free series reads ~0.
- 3 new tests.

## 0.36.0 - variance-ratio test

- `stats.py` gains `variance_ratio` (Lo-MacKinlay): the per-period variance of
  q-step moves over 1-step moves, ~1 for a random walk, above 1 when trending,
  below 1 when mean-reverting. Verified across all three regimes.
- 4 new tests.

## 0.35.0 - Hurst exponent

- `quant_system/stats.py`: `hurst_exponent` estimates long-memory from the scaling
  of lagged-difference dispersion. Tested across the three regimes: a random walk
  reads ~0.5, positively-autocorrelated increments read above 0.5 (persistent),
  and an OU-like series reads below 0.5 (mean-reverting).
- 3 new tests.

## 0.34.0 - bet sizing from probabilities

- `signals/bet_sizing.py`: `bet_size` turns a classifier's probability that its
  side is correct into a size in [-1, 1] via the Lopez de Prado map, a coin flip
  sizes to zero and near-certainty to a full bet; `discretize_bets` rounds sizes
  to a step so small probability wiggles do not churn the book.
- 6 new tests.

## 0.33.0 - risk-budgeting allocators

- `portfolio/risk_budget.py`: `erc_weights` solves for equal-risk-contribution
  (true risk parity) weights, which reduce to inverse-vol only when assets are
  uncorrelated; `risk_contributions` reports each asset's share of portfolio
  variance; and `max_diversification_weights` maximises the diversification ratio,
  tilting toward assets that hedge rather than merely quiet ones.
- 5 new tests.

## 0.32.0 - sequential bootstrap

- `signals/sampling.py`: the sequential bootstrap (Lopez de Prado, ch. 4) for
  overlapping labels. `indicator_matrix` marks which bars each label's window
  spans, `average_uniqueness` averages 1/concurrency over a label's bars, and
  `sequential_bootstrap` draws samples one at a time, weighting each pick by its
  average uniqueness given the draws so far, so the resample overlaps less than a
  uniform one.
- Verified on the canonical case: two identical labels and one disjoint label give
  uniqueness [0.5, 0.5, 1.0], and the sequential bootstrap draws the disjoint label
  about 40% of the time against a uniform 33%.
- 5 new tests (332 total).

## 0.31.0 - CUSUM event filter

- `signals/labeling.py` gains `cusum_events`, the symmetric CUSUM filter (Lopez
  de Prado, ch. 2). Running positive and negative sums accumulate the bar-to-bar
  moves and reset whenever one crosses a threshold, so the returned positions are
  the bars where the cumulative move broke that threshold, up or down. Passing log
  prices makes the threshold a cumulative-return level.
- This is the event sampler that feeds `triple_barrier_labels`: instead of
  labeling every bar, label the ones where something happened. A test runs the
  full pipeline (CUSUM events -> barrier labels) end to end.
- 6 new tests (327 total).

## 0.30.0 - triple-barrier labeling

- `signals/labeling.py`: triple-barrier labels (Lopez de Prado). Each event is
  labeled by the first barrier its forward path touches, a volatility-scaled
  profit target (+1), a stop (-1), or a holding-period limit (the sign of the
  return at the limit), so the label reflects the path a trade would have taken
  rather than only its endpoint. `ewm_volatility` sets the barrier widths.
- Barrier widths scale with volatility, so a target means the same in a calm and
  a wild market; disabling a horizontal barrier (pt or sl of 0) is supported, and
  events without a volatility estimate are skipped.
- 6 new tests (321 total).

## 0.29.0 - paper-book repricing harness

- `options/monitor.py`: `repricing_health` grades a built surface pass/fail, so a
  scheduled job can refuse bad data. It fails on an empty surface, an implied-vol
  failure rate above a threshold, or arbitrage that exceeds the local bid-ask
  spread (the violations that usually mean stale data, not mid-price artefacts).
- `scripts/reprice_paper_book.py` pulls a live chain, rebuilds the surface, fits
  the arbitrage-free SSVI surface, writes a JSON run log, and exits non-zero when
  the data is unhealthy. It does not trade and does not commit anything.
- `.github/workflows/paper-book.yml` runs the repricer and uploads the log as a
  build artifact. Two deliberate restraints: the daily schedule is committed but
  commented out (manual-dispatch only until someone opts in), and the log is an
  artifact rather than a repo commit, so the job never writes automated commits.
- 6 new tests (315 total).

## 0.28.0 - fractional differentiation

- `signals/frac_diff.py`: fixed-width fractional differentiation (Lopez de
  Prado). `ffd_weights` builds the truncated binomial weights (they reduce to
  `[1, -1]` at d=1 and `[1]` at d=0), `frac_diff_ffd` applies them, and
  `min_ffd_order` searches for the smallest order whose series passes an
  augmented Dickey-Fuller test.
- The point, shown in the tests: a random walk needs about d=0.3 to become
  stationary, and at that order it stays ~0.9 correlated with the price level,
  where a full first difference (d=1) keeps almost none of the memory. That is
  the trade fractional differencing is for, stationarity without amnesia.
- 7 new tests (309 total).

## 0.27.0 - shared research helpers (refactor)

- `quant_system/research.py` holds the setup the result scripts all repeated: the
  standard ticker `research_universe`, and `sleeve_makers`, which returns the
  three canonical sleeves (momentum, pairs, ML) as walk-forward weight callbacks
  with optional regime sizing. The seven `build_*_results.py` scripts now import
  these instead of each carrying their own copy, so "the three sleeves" is defined
  in one place.
- Behaviour is unchanged: the extracted closures are verbatim, `sleeve_makers` is
  tested on synthetic data (each sleeve returns weights on the panel index, and
  the defensive-regime scaling halves the directional books), and rerunning the
  blend script reproduced its README numbers within the usual yfinance drift.
- Imports left unused by the migration were removed from the scripts.
- 4 new tests (302 total).

## 0.26.0 - SSVI surface fit against a live chain

- `options/ssvi.py` gains `ssvi_surface_calendar_free`, the direct calendar
  check: total variance non-decreasing in maturity at every log-moneyness. Unlike
  the sufficient parameter conditions in `ssvi_surface_arbitrage_free`, this is
  the definition, so it can certify a surface the sufficient bound fails to.
- `scripts/build_ssvi_surface.py` fits one arbitrage-free SSVI surface to a real
  chain (live by default, synthetic with a flag) and overlays it per expiry,
  reporting both the sufficient-condition and the direct pointwise verdicts.
- The live SPY figure (`docs/results/ssvi_surface_spy.png`) shows two honest
  things: the single arbitrage-free surface fits looser than the per-expiry raw
  SVI, which is the cost of one skew and one curvature law spanning the whole
  surface; and the sufficient calendar condition reads False while the direct
  check reads True, because the steep front-month curvature sits on the sufficient
  bound, a concrete case of the conditions being sufficient and not necessary.
- 3 new tests (298 total).

## 0.25.0 - calendar-arbitrage-free SSVI surface

- `options/ssvi.py` gains a full-surface fit. `SSVISurface` holds one `rho`, one
  power-law curvature `phi(theta) = eta * theta^-gamma`, and an at-the-money
  total variance per maturity; `ssvi_surface_slice` reads off each expiry's slice
  and `ssvi_surface_arbitrage_free` returns the (butterfly, calendar) verdict.
- The calendar condition is the natural one in total-variance coordinates: the
  at-the-money variance is non-decreasing in maturity, plus the SSVI bound on
  `d/dtheta (theta*phi)`. `fit_ssvi_surface` fits `rho, eta, gamma` and the theta
  term structure jointly, building theta from non-negative increments so it is
  monotone by construction, and penalising the butterfly and phi bounds, so the
  fitted surface is provably free of both butterfly and calendar arbitrage.
- On a surface generated from known parameters the fit recovers them to ~1e-10;
  fed a term structure whose variance falls with maturity (a calendar arbitrage),
  it returns a monotone, calendar-free surface at a cost in fit error rather than
  reproducing the violation.
- This completes the arbitrage-free surface the SSVI slice work (0.24) pointed at.
  The conditions remain sufficient, not necessary, which the code states.
- 7 new tests (295 total).

## 0.24.0 - SSVI arbitrage-free surface parameterization

- `options/ssvi.py`: the SSVI slice (Gatheral and Jacquier 2014), a three-
  parameter (theta, rho, psi) sub-family of raw SVI whose parameters carry
  *sufficient* conditions for no butterfly arbitrage. `ssvi_total_variance`
  evaluates it, `ssvi_to_svi_params` maps it to the equivalent raw SVI exactly (so
  the g(k) check and plotting reuse), and `ssvi_butterfly_free` tests the two
  conditions `theta*psi*(1+|rho|) < 4` and `theta*psi^2*(1+|rho|) <= 4`.
- `fit_ssvi_slice` calibrates a slice while penalising those quantities past
  their bound, so the fit stays inside the arbitrage-free region. Where the
  penalised raw-SVI fit (0.23) drives g(k) toward the boundary as a soft
  constraint, an SSVI fit that satisfies the conditions is provably clean: fitting
  it to an arbitrageable smile returns a slice that both the conditions and the
  independent g(k) check confirm is arbitrage-free, at a cost in RMSE.
- The conditions are sufficient, not necessary, which the code states; the
  calendar-arbitrage side (a monotone theta term structure) is the natural
  extension to a full arbitrage-free surface and is left for a focused follow-up.
- 9 new tests (288 total).

## 0.23.0 - arbitrage-free SVI calibration

- `options/svi.py` gains `fit_svi_slice_no_arb`, an SVI calibration that adds a
  penalty on Gatheral's g(k) falling below a small positive margin over a padded
  grid, so the optimiser trades a little data fit for a slice whose implied
  density stays positive. On a smile generated from arbitrageable parameters the
  unconstrained fit reproduces the violation (min g -5.9) while the penalised fit
  is clean (min g +0.001) at the cost of RMSE; on an already-clean smile it pays
  no penalty. It is a strong soft constraint, not a certificate of positivity,
  and the docstring says so.
- `fit_svi_points` and `fit_svi_surface` take an `arbitrage_free` flag, and
  `scripts/build_svi_fit.py` takes `--arb-free`, so the whole-surface fit and the
  figure can be produced with the butterfly penalty on.
- This directly answers the "future work" left by 0.22: the short-dated SPY
  slices that came back arbitrageable under the plain fit can now be fit
  arbitrage-free.
- 4 new tests (279 total).

## 0.22.0 - SVI fit against a live market smile

- `options/svi.py` gains `svi_implied_vol`, converting a fitted slice's total
  variance back to a Black-Scholes vol so it can be plotted against quoted vols.
- `scripts/build_svi_fit.py` fits SVI to a real surface (a live chain by default,
  synthetic with a flag) and overlays the fits on the market smiles, annotated
  with the RMS error and the butterfly no-arbitrage verdict per expiry.
- The live SPY figure (`docs/results/svi_fit_spy.png`) shows the honest picture:
  SVI tracks the smiles to a fraction of a vol point, but the short-dated slices
  are flagged not arbitrage-free because they reproduce butterfly violations
  already present in the raw quotes, while the longer expiries fit cleanly. A
  visually perfect fit can still be arbitrageable, which is why g(k) is checked
  rather than assumed; a genuinely arbitrage-free calibration (constraining the
  fit to keep g >= 0) is noted as future work.
- 2 new tests (275 total).

## 0.21.0 - HRP allocation across strategy sleeves

- `portfolio/allocator.py` gains `hrp_allocations`: a causal rolling
  Hierarchical Risk Parity allocation across the sleeves, the correlation-aware
  counterpart to `inverse_vol_allocations`. It refits `hrp_weights` on the
  trailing window every 21 days and holds between refits; every row is fit on
  returns strictly before its day.
- `scripts/build_hrp_blend_results.py` combines the three sleeves into one
  10%-vol book three ways (inverse-vol, HRP, equal weight) and scores them on the
  same window. The window alignment matters: HRP warms up longer than
  inverse-vol, and scoring them over different date ranges flipped the
  inverse-vol Sharpe from -0.38 to 0.09, so all books are trimmed to a common
  index before metrics.
- The finding: both risk-based methods beat naive equal weight (which loads the
  two losing sleeves), and HRP edges out inverse-vol (0.26 vs 0.09 Sharpe). Both
  realize well under the 10% target because they concentrate into the low-vol
  pairs sleeve and the leverage cap binds, which the README explains rather than
  papers over.
- 3 new tests (273 total).

## 0.20.0 - realized-vol vs GARCH regime sizing comparison

- `scripts/build_regime_results.py` wires the GARCH regime (added in 0.15) into a
  real backtest for the first time, comparing three defensive-sizing overlays on
  the sector-momentum sleeve: none, the realized-vol-ratio regime, and the GARCH
  regime. The finding is that the forecast-based detector does not win: it reaches
  the lowest average volatility but turns risk down at the wrong moments, so it
  prints a worse Sharpe (-0.25) and a deeper drawdown (-30.3%) than the slow
  realized-vol detector (0.00 Sharpe, -20.8% drawdown) or doing nothing. The base
  sleeve is near-zero Sharpe, so this is a narrow test on a weak signal, stated as
  such in the README, but the direction is a useful check on the assumption that a
  more sophisticated volatility model sizes better.
- No library code changed; this is an analysis built on the existing, tested
  regime and backtest code (test count unchanged at 270).

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
