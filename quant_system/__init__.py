"""quant_system - a small but technically correct quantitative research stack.

The package is organised into independent layers so that each piece can be
reasoned about on its own:

    data/         price/volume loading with on-disk caching + offline fallback
    signals/      strategy logic that turns prices into *target weights*
    risk/         position sizing, portfolio limits, tail-risk metrics
    regime/       market-regime detection and signal routing
    backtest/     the event-driven engine, transaction costs, walk-forward
    performance/  analytics, tearsheet, Fama-French factor decomposition
    monitor/      out-of-band monitors (SEC EDGAR, LLM analyst)

Design rules that hold across the whole package:
  * No global mutable state. Functions take inputs and return outputs.
  * The 1-day execution lag lives in exactly one place (the backtest engine),
    so there is a single source of truth for look-ahead prevention.
  * Everything that touches a parameter reads it from :mod:`quant_system.config`.
"""

__version__ = "0.5.0"
