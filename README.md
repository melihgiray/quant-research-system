# quant-research-system

A backtesting setup for a few systematic equity strategies on daily data. It runs
three strategies through one engine, charges realistic trading costs, tests them
out of sample, and prints the results. Nothing here connects to a broker and
nothing trades live. I built it to find out whether these ideas survive once
costs and proper out-of-sample testing get in the way.

## Results

Walk-forward out-of-sample, real daily data (yfinance), 2016-01-04 to
2024-10-07, net of spread and square-root impact costs at $1M capital.
Reproduce with `python scripts/build_results.py`.

| Strategy | OOS Sharpe | Ann. return | Max drawdown | Turnover |
|---|---|---|---|---|
| Cross-sectional momentum (11 sector ETFs) | 0.00 | -0.4% | -20.8% | 7.6x |
| Pairs trade (per-fold selection) | 0.42 | +0.5% | -1.6% | 1.1x |
| ML directional (24 large caps) | -0.82 | -3.4% | -27.7% | 112.7x |

![Out-of-sample equity curves](docs/results/equity_oos.png)

Reading these honestly: none of the three clears a significance bar after
costs, and the ML signal's edge is eaten alive by its own 112x turnover. That
is the finding, not a failure of the plumbing. The framework's job is to stop
a weak signal from looking strong, and two details in this table show it
working. First, sector momentum's Sharpe of 0.00 is what the real
Jegadeesh-Titman effect tends to look like on a small, recent, cost-charged
universe. Second, the pairs number used to be better: selecting the pair on
the full sample (which peeks at the test years) gave a 0.61 Sharpe, and
re-selecting the pair inside each walk-forward fold, which is the only thing
you could actually do in real time, deflates it to 0.42. The gap between those
two numbers is measured look-ahead bias.

## What's in it

Three strategies:

- **Cross-sectional momentum.** Rank the universe by its 12-month return skipping
  the most recent month, go long the top names and short the bottom, rebalance
  monthly. The last month gets skipped because short-term returns tend to reverse.
- **A pairs trade.** Find a cointegrated pair with the Engle-Granger test, trade
  the spread when it stretches past two standard deviations, and close it when it
  reverts. If the cointegration breaks down (p-value drifts above 0.10) it stops
  trading that pair. The pair is re-selected inside each walk-forward fold using
  only data available at the fold boundary, with a Benjamini-Hochberg correction
  across the candidates, so the choice of what to trade is as causal as the
  trading itself.
- **A machine-learning signal.** A gradient-boosted classifier predicts whether
  each name is up or down tomorrow from eight lagged features. The predicted
  probability sets the position size, so a confident call gets a bigger bet. SHAP
  values show which features it actually leans on, and the classifier can be
  scored with purged, embargoed cross-validation (`--cv`) so overlapping labels
  can't leak between the folds.

Around those:

- An engine that lags every signal by a day, so a position can never use a price
  it would not have had yet. The lag lives in one line, which makes it easy to
  check. There is a test that feeds the engine a deliberately cheating signal and
  confirms the lag strips its edge back to noise.
- Trading costs as a half-spread plus square-root market impact. Impact grows with
  how much you trade relative to daily volume, so the same strategy costs more at
  larger size. An optional participation cap (`--max-participation 0.05`) limits
  daily trading in a name to a fraction of its volume and carries the unfilled
  remainder to later days, the way a desk would actually work a large order.
- Walk-forward validation. The model refits on a rolling window and only the
  out-of-sample stretches get stitched into the final equity curve.
- The usual performance stats (Sharpe, Sortino, Calmar, drawdown, hit rate,
  turnover, VaR, CVaR), plus a Fama-French three-factor regression to see how much
  of the return is just market, size, and value exposure.
- A few honesty checks, because one number oversells. Probabilistic and Deflated
  Sharpe, bootstrap confidence intervals, a rough capacity estimate for how much
  capital the strategy could hold before costs eat it, and a Benjamini-Hochberg
  correction on the pairs scan so a pair picked from several candidates has to
  survive the fact that several were tried.
- Two side tools: a SEC EDGAR filing watcher that can text or call you when a
  company files paperwork to issue new shares, and a small script that asks Claude
  to turn a market snapshot into structured JSON.

## Running it

Needs Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# offline demo on generated data, no network needed
python -m quant_system.cli --synthetic

# real data from yfinance, cached locally after the first pull
python -m quant_system.cli --strategy all --universe all
```

Flags worth knowing: `--strategy {momentum,pairs,ml,all}`, `--synthetic`,
`--no-walk-forward`, `--bootstrap`, `--capacity`, `--shap`, `--rf 0.04`. Run
`pytest` for the test suite.

One note on data. yfinance rate-limits hard, and Yahoo hands back 429s if you pull
too much too fast. Prices get cached after the first good pull. If a pull fails,
the loader falls back to a generated dataset so the whole thing still runs end to
end. Anything from that fallback is labeled synthetic and is not real market data.

## A few design choices

The one-day lag. A signal formed at the close of day T is only allowed to earn day
T+1's return. The engine does this with a single `weights.shift(1)` rather than
scattering shifts through every signal, where one is easy to forget. Centralizing
it means there is exactly one place to get right and one place to test.

Square-root costs. Market impact grows with the square root of how much you trade
relative to daily volume, not linearly. Liquidity refills as you go, so the
marginal share is cheaper than the first. The practical consequence is that cost
depends on capital, which is what the capacity estimate uses.

The HMM is for looking, not sizing. The regime HMM decodes with Viterbi, which
reads the whole series including the future, so its labels are not safe to trade
on. Sizing runs off a causal volatility-ratio rule instead, and the HMM only feeds
the regime chart.

Synthetic data is honest about itself. The generated data has no real edge baked
in, so the strategies score near zero on it. That is the point. It exercises the
plumbing without pretending to be a result.

## Layout

```
quant_system/
  data/         loading, caching, the universe lists
  signals/      momentum, pairs, the ML signal
  risk/         sizing, drawdown and concentration limits, VaR/CVaR
  regime/       vol-ratio and HMM detection, regime-based sizing
  backtest/     the engine, cost model, walk-forward, capacity
  performance/  metrics, tearsheet, factor regression, significance, bootstrap
  monitor/      EDGAR watcher, Claude analyst
  cli.py        the entry point
tests/          unit tests
```

## Limitations

The factor proxies are ETFs (SPY, IWM, VTV, VUG), not the Ken French research
factors. Fine for a sanity check, swappable if I want the real thing. The
candidate-pair list itself is still hand-picked with hindsight (I chose pairs
known to move together), even though selection among them is now causal per
fold. Universe membership is today's list, so there is survivorship bias in the
single names. And none of this has traded a real dollar.

## License

MIT.
