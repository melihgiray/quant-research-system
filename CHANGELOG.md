# Changelog

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
