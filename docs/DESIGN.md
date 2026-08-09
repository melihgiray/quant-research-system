# Design

The choices that shaped this repo, in one place. The theme running through all
of them is that a backtest's job is to talk you out of bad ideas, not to flatter
good-looking ones, so most of the design effort went into removing the ways a
result can lie.

## No look-ahead, anywhere

Every number that feeds a decision uses only data that existed at decision time.

- **Walk-forward, not a single split.** `backtest/walk_forward.py` rolls an
  in-sample window, an out-of-sample window and a step forward through time, so
  the reported metrics are stitched from many out-of-sample slices rather than
  one lucky test set.
- **Selection is causal too.** It is not enough to trade causally if you *chose*
  what to trade with hindsight. The pairs strategy re-runs cointegration
  selection inside each fold on data available at the fold boundary
  (`signals/mean_reversion.py`), and the README reports the gap between that and
  full-sample selection as a measured quantity: full-sample selection scored a
  0.61 Sharpe, per-fold 0.42, and the difference is look-ahead bias you can put
  a number on.
- **Estimates are lagged.** Trailing volatility, sleeve allocations and the
  vol-target scaler are all shifted a day (`portfolio/allocator.py`,
  `risk/sizing.py`), so the weight worn on day *t* depends only on returns
  through *t-1*.

## Costs are charged before anything is believed

`backtest/costs.py` charges a bid-ask spread plus a square-root market-impact
term sized to capital, and `backtest/capacity.py` sweeps that cost against
capital so a strategy that only works at toy size is exposed as such. The
headline tables are all net of these frictions. The interesting output is
usually how much of an apparent edge the costs eat: the ML sleeve's raw signal,
for instance, is undone by its own 112x turnover, and that is reported as the
finding rather than hidden.

## Guarding against overfitting and multiple testing

A framework that lets you try many strategies must correct for the fact that you
tried many strategies.

- **Deflated and probabilistic Sharpe** (`performance/significance.py`) adjust
  the Sharpe ratio for track-record length, skew, kurtosis and the number of
  trials, and report a minimum track-record length.
- **Bootstrap confidence intervals** (`performance/bootstrap.py`) put error bars
  on Sharpe and CAGR instead of a single point estimate.
- **Benjamini-Hochberg FDR** (`performance/multiple_testing.py`) controls the
  false-discovery rate across the pairs cointegration scan and across the ML
  feature search, so a feature that looks significant on its own must survive
  being one of many tests. The ML search uses permutation-null p-values
  (`signals/feature_selection.py`), and the demonstration is deliberately
  humbling: only one of eight features survives the correction.

## Risk-based allocation over mean-variance

Weights come from risk structure, not from inverting a noisy covariance matrix,
which is where classic mean-variance optimisation blows up.

- **Inverse-vol blending with a volatility target** (`portfolio/allocator.py`)
  runs the strategy sleeves as one book at a chosen annual volatility.
- **Hierarchical Risk Parity** (`portfolio/hrp.py`) clusters assets on their
  correlation, quasi-diagonalises, and splits the risk budget down the tree
  without inverting anything. The comparison script shows it earning the best
  risk-adjusted return of the allocators tested, and the write-up is careful to
  report the counter-intuitive part (it is more concentrated by name, not less).

## Pure functions, tested at the convention

Signals, the engine, analytics and the allocators are separated so each is a
pure function of its inputs and testable in isolation. Tests pin conventions,
not implementations: a test fails if the reported metrics stop being causal, or
if an allocator stops summing to one, rather than asserting an exact internal
call sequence. CI (`.github/workflows/ci.yml`) runs the suite on every push.

## Refuse to invent data

The results scripts abort rather than publish numbers built on synthetic prices
(`load_price_data(..., allow_synthetic_fallback=False)`), and every headline is
reproducible by a single command named in the README. Where the model or the
data did not provide something, it is left blank rather than filled with a
plausible guess; the options leg, for example, carries vendor Greeks through as
NaN when they are absent instead of back-solving a substitute.

## What this is not

It does not connect to a broker, it does not trade, and it makes no claim that
any strategy here is profitable after costs. Several are not, and that is stated
plainly. The value is the measurement apparatus and the honesty of what it
reports, which is the thing worth showing.
