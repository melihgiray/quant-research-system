# Roadmap

Things I want to add. Roughly in priority order, grouped by area. A lot of the
evaluation work follows the López de Prado and Bailey papers on how backtests
mislead.

Done so far is marked `[x]`.

## Knowing whether the edge is real
- [x] Probabilistic and deflated Sharpe ratio, plus minimum track record length.
- [x] Bootstrap confidence intervals for Sharpe and CAGR.
- [x] Capacity / cost-sensitivity sweep over capital.
- [ ] Multiple-testing control (Benjamini-Hochberg) on the pairs and feature search.

## Realism and portfolio construction
- [ ] Better execution: participation cap, next-open fills, partial fills.
- [ ] Combine the three strategies into one book with volatility targeting.
- [ ] Hierarchical risk parity allocator, compared against inverse-variance.

## Machine learning
- [ ] Purged and embargoed cross-validation to stop label leakage.
- [ ] A proper sklearn pipeline with calibration and sample weights by uniqueness.
- [ ] Meta-labeling: a second model decides whether and how big to act.

## Volatility and risk
- [ ] GARCH(1,1) conditional-vol forecasts for sizing.
- [ ] Cornish-Fisher and EVT tail estimates, plus a couple of stress scenarios.

## Reporting and tooling
- [ ] A single HTML report with rolling Sharpe/beta and a per-year table.
- [ ] CI running the test suite, and a short design write-up.

Each change comes with a test, and the suite stays green.
