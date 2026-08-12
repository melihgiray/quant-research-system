# Roadmap

Things I want to add. Roughly in priority order, grouped by area. A lot of the
evaluation work follows the López de Prado and Bailey papers on how backtests
mislead.

Done so far is marked `[x]`.

## How I work on this

I build in small steps and commit each one on its own, then push. A new helper
and its test are a commit. Wiring that helper into the CLI is the next commit.
The docstring pass and the changelog entry are their own commits. So each item
below is written as an ordered list of commit-sized steps, and a single item
usually turns into three to six commits rather than one.

Two rules keep that honest, which matters because the whole point of this repo
is that its history holds up to scrutiny:

- Every commit builds and passes `pytest` on its own. Any commit can become the
  version someone checks out, so none of them are left broken.
- No padding. No empty commits, no whitespace-only commits, no splitting one
  indivisible change into pieces that do not stand alone. The frequency comes
  from genuinely working in small increments, not from inflating the count.

The per-item steps below are the default decomposition. When a step turns out to
be bigger than expected, it splits further; when two are trivially entangled,
they merge. The list is a guide, not a quota.

## Knowing whether the edge is real
- [x] Probabilistic and deflated Sharpe ratio, plus minimum track record length.
- [x] Bootstrap confidence intervals for Sharpe and CAGR.
- [x] Capacity / cost-sensitivity sweep over capital.
- [x] Multiple-testing control (Benjamini-Hochberg) on the pairs scan.
- [x] Extend the FDR control to the ML feature search.
  1. Per-feature importance p-values (permutation null) + test.
  2. Apply the existing Benjamini-Hochberg pass across features + test.
  3. Surface surviving features in the ML summary and CLI (`--feature-fdr`).
  4. Docstrings, changelog, README numbers.

## Realism and portfolio construction
- [x] Volume-participation cap with partial fills that carry over days.
- [ ] Next-open fills (needs open prices in the loader).
  1. Carry the Open column through the loader and cache + test.
  2. Add a next-open fill mode to the engine + test.
  3. Config flag and CLI wiring.
  4. Docs and changelog.
- [x] Combine the three strategies into one book with volatility targeting.
  1. Inverse-vol / equal-risk allocator as a pure function + test.
  2. Stitch the three sleeves into one weight matrix + test.
  3. Vol-target the combined book through `risk/sizing` + test.
  4. A results script + a README row for the blended book.
  5. Docs and changelog.
- [x] Hierarchical risk parity allocator, compared against inverse-variance.
  1. Correlation-distance and tree clustering helper + test.
  2. Recursive bisection allocation + test.
  3. A script comparing HRP against inverse-variance, with a plot.
  4. Docs and changelog.

## Machine learning
- [x] Purged and embargoed cross-validation to stop label leakage.
- [x] A proper sklearn pipeline with calibration and sample weights by uniqueness.
  1. Wrap the model in an sklearn `Pipeline` + test.
  2. Probability calibration + test.
  3. Sample-uniqueness (label concurrency) weights + test.
  4. Route `train_predict` through the pipeline.
  5. Docs and changelog.
- [ ] Meta-labeling: a second model decides whether and how big to act.
  1. Extract the primary signal's side + test.
  2. Train the secondary model on meta-labels + test.
  3. Size from the meta-probability + test.
  4. Backtest wiring and a README row.
  5. Docs and changelog.

## Volatility and risk
- [x] GARCH(1,1) conditional-vol forecasts for sizing (the `arch` package is installed).
  1. A GARCH wrapper returning a one-step conditional-vol forecast + test.
  2. A causal (lagged) forecast series over a price path + test.
  3. Feed it into sizing and expose it as a third regime definition + test.
  4. Docs and changelog.
- [ ] Cornish-Fisher and EVT tail estimates, plus a couple of stress scenarios.
  1. Cornish-Fisher VaR + test.
  2. EVT peaks-over-threshold tail + test.
  3. Named stress scenarios (2008, 2020 shocks) + test.
  4. Surface them in the tearsheet.
  5. Docs and changelog.

## Reporting and tooling
- [x] A single HTML report with rolling Sharpe/beta and a per-year table.
  1. A self-contained HTML tearsheet builder + test.
  2. Rolling Sharpe/beta plot and a per-year return table.
  3. A script that writes the report.
  4. Docs and changelog.
- [x] CI running the test suite, and a short design write-up.
  1. A GitHub Actions workflow that runs `pytest` on push.
  2. A status badge in the README.
  3. `docs/DESIGN.md` covering the main choices.

## Options leg
Done: Black-Scholes and CRR pricing with cross-checked Greeks; a live vol surface
with graded no-arbitrage checks; covered-call / cash-secured-put / delta-hedged
short-straddle backtests; a real historical-chain loader validated against vendor
IVs. Next:
- [ ] Re-engineer a licensed research codebase (an SVI surface fit is the natural fit).
  1. Record the source, author, commit SHA and license verbatim.
  2. Port the algorithm into a typed module skeleton.
  3. Fill in the rewrite with config instead of constants + tests.
  4. A profiled before/after, reproducible by a script.
  5. Attribution docs and changelog.
- [ ] Production harness: a scheduled job that repriced a small paper book.
  1. A GitHub Actions cron workflow.
  2. The paper-book repricer that pulls the live chain and rebuilds the surface.
  3. Fail-loud checks on bad data (empty chain, all-IV-failures, arb over spread).
  4. Persist run logs in the repo.
  5. A README section: what runs, when, and what it does not do.
