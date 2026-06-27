"""Market-regime detection and regime-aware signal routing."""

from .detector import (
    RegimeResult,
    detect_regime,
    vol_ratio_regime,
    hmm_regime,
    market_proxy_returns,
)
from .switcher import apply_regime_sizing, switch_strategies

__all__ = [
    "RegimeResult",
    "detect_regime",
    "vol_ratio_regime",
    "hmm_regime",
    "market_proxy_returns",
    "apply_regime_sizing",
    "switch_strategies",
]
