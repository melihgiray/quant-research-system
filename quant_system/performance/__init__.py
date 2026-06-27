"""Performance analytics, tearsheet, and Fama-French factor decomposition."""

from .analytics import compute_metrics, sharpe_ratio, sortino_ratio, calmar_ratio
from .tearsheet import format_tearsheet, save_report_plots
from .factor_decomp import factor_decomposition, FactorDecompResult
from .significance import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    min_track_record_length,
    significance_summary,
)
from .bootstrap import (
    sharpe_confidence_interval,
    cagr_confidence_interval,
    bootstrap_summary,
    ConfidenceInterval,
)

__all__ = [
    "compute_metrics",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "format_tearsheet",
    "save_report_plots",
    "factor_decomposition",
    "FactorDecompResult",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "min_track_record_length",
    "significance_summary",
    "sharpe_confidence_interval",
    "cagr_confidence_interval",
    "bootstrap_summary",
    "ConfidenceInterval",
]
