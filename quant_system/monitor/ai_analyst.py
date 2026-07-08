"""LLM market analyst with structured JSON output and alert thresholds.

Sends a market snapshot to Claude and gets back something a system can act on
instead of free text:

  * STRUCTURED OUTPUT - the model is forced to call a tool whose input_schema is
    our analysis schema, so we get back validated JSON (sentiment, risk score,
    risks, action) instead of prose we would have to regex. No parsing, no
    "the model added a preamble" failures.
  * PROMPT CACHING - the long, static analyst instructions are sent with
    cache_control so repeated polls reuse the cached system prompt (cheaper +
    lower latency on a monitor that runs every few minutes).
  * ALERT THRESHOLDS - a pure ``should_alert`` function turns the structured
    assessment into a yes/no alert decision, kept separate from the model call so
    it is testable without the network.

The anthropic dependency is imported lazily so the rest of the package works
without it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Default to a current Claude model; override via AnalystConfig.model.
DEFAULT_MODEL = "claude-sonnet-4-6"

# JSON schema the model must fill in (also the tool input_schema).
ANALYSIS_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["bullish", "neutral", "bearish"],
            "description": "Directional read over the next few trading days.",
        },
        "risk_score": {
            "type": "number",
            "description": "Overall risk from 0 (calm) to 1 (acute stress).",
        },
        "confidence": {
            "type": "number",
            "description": "Model confidence in this assessment, 0 to 1.",
        },
        "key_risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short bullet list of the most material risks.",
        },
        "recommended_action": {
            "type": "string",
            "enum": ["accumulate", "hold", "reduce", "hedge", "exit"],
            "description": "Single suggested portfolio action.",
        },
        "rationale": {
            "type": "string",
            "description": "Two or three sentences justifying the assessment.",
        },
    },
    "required": ["sentiment", "risk_score", "confidence", "key_risks",
                 "recommended_action", "rationale"],
}


@dataclass
class AnalystConfig:
    """Configuration for the analyst.

    model:            Claude model id.
    max_tokens:       response cap.
    alert_risk_score: emit an alert if risk_score >= this.
    alert_sentiment:  sentiments that should always alert.
    alert_actions:    recommended actions that should always alert.
    api_key:          Anthropic key (defaults to ANTHROPIC_API_KEY env var).
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 1024
    alert_risk_score: float = 0.7
    alert_sentiment: tuple = ("bearish",)
    alert_actions: tuple = ("reduce", "hedge", "exit")
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))


def _system_prompt() -> str:
    """Static analyst instructions - long and stable, so we cache it."""
    return (
        "You are a disciplined sell-side risk analyst embedded in a quantitative "
        "trading desk. You are given a snapshot of market data for one or more "
        "instruments. Assess the near-term (1-5 day) risk and direction.\n"
        "Rules:\n"
        " - Be calibrated, not dramatic. Most days are 'neutral' with low risk.\n"
        " - Base the assessment ONLY on the provided data; do not invent news.\n"
        " - risk_score reflects volatility, drawdown and dislocation in the data.\n"
        " - Always return your answer by calling the `record_assessment` tool.\n"
        " - Keep rationale concise and falsifiable."
    )


def _snapshot_to_text(snapshot: Dict) -> str:
    """Render a market-data snapshot dict into a compact prompt block."""
    lines = ["Market snapshot:"]
    for ticker, fields in snapshot.items():
        parts = ", ".join(f"{k}={v}" for k, v in fields.items())
        lines.append(f"  {ticker}: {parts}")
    return "\n".join(lines)


def analyze_snapshot(snapshot: Dict, cfg: AnalystConfig = None, client=None) -> Dict:
    """Send a market snapshot to Claude and return a validated assessment dict.

    Parameters
    ----------
    snapshot : dict
        {ticker: {field: value}}, e.g. {"SPY": {"price": 500, "change_pct": -2.1,
        "vol_21d": 0.18, "rsi": 28}}.
    cfg : AnalystConfig
        Model + threshold configuration.
    client : anthropic.Anthropic, optional
        Inject a client (e.g. for testing). Otherwise one is constructed from the
        configured API key.

    Returns
    -------
    dict
        Structured assessment matching ANALYSIS_SCHEMA.

    Raises
    ------
    RuntimeError
        If the anthropic SDK is unavailable or no API key is configured.
    """
    cfg = cfg or AnalystConfig()
    if client is None:
        try:
            import anthropic
        except Exception as exc:
            raise RuntimeError(f"anthropic SDK not installed: {exc}")
        if not cfg.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=cfg.api_key)

    tools = [{
        "name": "record_assessment",
        "description": "Record the structured market risk assessment.",
        "input_schema": ANALYSIS_SCHEMA,
    }]

    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        # cache_control on the static system block -> cached across polls.
        system=[{"type": "text", "text": _system_prompt(),
                 "cache_control": {"type": "ephemeral"}}],
        tools=tools,
        tool_choice={"type": "tool", "name": "record_assessment"},
        messages=[{"role": "user", "content": _snapshot_to_text(snapshot)}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_assessment":
            return dict(block.input)
    raise RuntimeError("model did not return a structured assessment")


def should_alert(analysis: Dict, cfg: AnalystConfig = None) -> Tuple[bool, List[str]]:
    """Decide whether a structured assessment warrants an alert.

    Pure function (no network) so the alerting policy is unit-testable. Returns
    (alert?, reasons).
    """
    cfg = cfg or AnalystConfig()
    reasons: List[str] = []
    if analysis.get("risk_score", 0.0) >= cfg.alert_risk_score:
        reasons.append(f"risk_score {analysis['risk_score']:.2f} >= {cfg.alert_risk_score}")
    if analysis.get("sentiment") in cfg.alert_sentiment:
        reasons.append(f"sentiment={analysis.get('sentiment')}")
    if analysis.get("recommended_action") in cfg.alert_actions:
        reasons.append(f"action={analysis.get('recommended_action')}")
    return (len(reasons) > 0, reasons)


def _main() -> int:
    """Demo: analyse a hard-coded stressed snapshot (needs ANTHROPIC_API_KEY)."""
    import argparse
    p = argparse.ArgumentParser(description="LLM market analyst (structured output)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    snapshot = {
        "SPY": {"price": 478.2, "change_pct": -2.6, "vol_21d": 0.21, "rsi_14": 27},
        "VIX": {"level": 31.5, "change_pct": 18.0},
    }
    cfg = AnalystConfig(model=args.model)
    try:
        analysis = analyze_snapshot(snapshot, cfg)
    except RuntimeError as exc:
        print(f"[ai_analyst] cannot run live ({exc}). Schema is still available offline:")
        print(json.dumps(ANALYSIS_SCHEMA, indent=2))
        return 0
    print(json.dumps(analysis, indent=2))
    alert, reasons = should_alert(analysis, cfg)
    print(f"ALERT: {alert} {reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
