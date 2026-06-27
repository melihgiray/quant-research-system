"""Out-of-band monitors: SEC EDGAR dilution watcher and an LLM market analyst.

These are operational tools, not part of the backtest. Both keep their heavy /
optional dependencies (twilio, anthropic) behind guarded imports so importing the
package never fails just because an optional extra is missing.
"""

from .edgar_enhanced import (
    EdgarMonitorConfig,
    resolve_cik,
    fetch_recent_filings,
    detect_dilution_filings,
    run_once,
    send_sms_alert,
    make_phone_call,
    dispatch_alert,
)
from .ai_analyst import AnalystConfig, analyze_snapshot, should_alert, ANALYSIS_SCHEMA

__all__ = [
    "EdgarMonitorConfig",
    "resolve_cik",
    "fetch_recent_filings",
    "detect_dilution_filings",
    "run_once",
    "send_sms_alert",
    "make_phone_call",
    "dispatch_alert",
    "AnalystConfig",
    "analyze_snapshot",
    "should_alert",
    "ANALYSIS_SCHEMA",
]
