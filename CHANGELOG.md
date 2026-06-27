# Changelog

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
