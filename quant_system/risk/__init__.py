"""Risk layer: position sizing, portfolio limits, and tail-risk metrics."""

from .sizing import fixed_fractional, vol_target_scale, kelly_weights
from .limits import apply_drawdown_stop, cap_concentration, check_limits
from .validation import (
    christoffersen_independence_test,
    christoffersen_conditional_coverage_test,
    basel_traffic_light,
    dynamic_quantile_test,
)
from .metrics import (
    historical_var,
    conditional_var,
    parametric_var,
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
)

__all__ = [
    "fixed_fractional",
    "vol_target_scale",
    "kelly_weights",
    "apply_drawdown_stop",
    "cap_concentration",
    "check_limits",
    "christoffersen_independence_test",
    "christoffersen_conditional_coverage_test",
    "basel_traffic_light",
    "dynamic_quantile_test",
    "historical_var",
    "conditional_var",
    "parametric_var",
    "drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
]
