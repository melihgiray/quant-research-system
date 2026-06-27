"""Configurable multi-ticker SEC EDGAR dilution monitor.

Watches a list of tickers and flags recently filed dilution-related forms: shelf
registrations and prospectus supplements (S-1, S-3, 424B5, 424B3), the paperwork
a company files before selling new shares. It polls SEC EDGAR's official JSON
submissions API. When something new shows up it can alert by text or phone call
through Twilio.

A few notes on how it is built:
  * It uses the documented data.sec.gov JSON API, not HTML scraping, with the
    descriptive User-Agent the SEC asks for.
  * Fetching and filtering are plain functions. The only stateful part is the
    polling loop, and even that takes its 'already seen' set as an argument.
  * No secrets in code. Twilio credentials come from environment variables.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import requests


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"

# Forms that typically precede or execute share dilution.
DEFAULT_DILUTION_FORMS = ("S-1", "S-3", "424B5", "424B3", "FWP")


@dataclass
class EdgarMonitorConfig:
    """Parameters for the monitor.

    tickers:        symbols to watch.
    forms:          filing types treated as dilution signals.
    lookback_days:  how far back a filing counts as "recent".
    user_agent:     SEC requires a descriptive UA like "Name email@example.com".
    poll_seconds:   delay between polling cycles in the run loop.
    alert_method:   how to notify on a new filing: "sms", "call", or "both".
                    A phone call is worth it for time-sensitive filings (a dilution
                    notice you want to see before the open), a text for the rest.
    """

    tickers: List[str] = field(default_factory=lambda: ["SPCE"])
    forms: tuple = DEFAULT_DILUTION_FORMS
    lookback_days: int = 7
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "SEC_USER_AGENT", "quant-research-system monitor (set SEC_USER_AGENT)"
        )
    )
    poll_seconds: int = 900
    alert_method: str = "sms"


def _sec_get(url: str, user_agent: str, timeout: int = 15) -> Optional[dict]:
    """GET a data.sec.gov JSON endpoint with the required UA header."""
    try:
        resp = requests.get(url, headers={"User-Agent": user_agent,
                                          "Accept-Encoding": "gzip, deflate"},
                            timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def load_ticker_cik_map(user_agent: str) -> Dict[str, str]:
    """Return {TICKER: zero-padded-10-digit CIK} from SEC's master mapping file."""
    data = _sec_get(SEC_TICKERS_URL, user_agent)
    out: Dict[str, str] = {}
    if not data:
        return out
    for row in data.values():
        out[str(row["ticker"]).upper()] = f"{int(row['cik_str']):010d}"
    return out


def resolve_cik(ticker: str, cik_map: Dict[str, str]) -> Optional[str]:
    """Map a ticker to its 10-digit CIK using a preloaded map."""
    return cik_map.get(ticker.upper())


def fetch_recent_filings(cik10: str, user_agent: str) -> List[dict]:
    """Return a flat list of the company's most recent filings.

    Each item: {form, filing_date, accession, primary_doc, url}. EDGAR returns
    the recent filings as parallel arrays, which we zip into records.
    """
    data = _sec_get(SEC_SUBMISSIONS_URL.format(cik10=cik10), user_agent)
    if not data:
        return []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accns = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    cik_int = int(cik10)
    out = []
    for i in range(len(forms)):
        accn = accns[i] if i < len(accns) else ""
        accn_nodash = accn.replace("-", "")
        doc = docs[i] if i < len(docs) else ""
        out.append({
            "form": forms[i],
            "filing_date": dates[i] if i < len(dates) else "",
            "accession": accn,
            "primary_doc": doc,
            "url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{doc}",
        })
    return out


def detect_dilution_filings(filings: List[dict], forms=DEFAULT_DILUTION_FORMS,
                            lookback_days: int = 7) -> List[dict]:
    """Filter filings to dilution-related forms within the lookback window."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
    forms_up = {f.upper() for f in forms}
    hits = []
    for f in filings:
        if f["form"].upper() not in forms_up:
            continue
        try:
            fdate = datetime.strptime(f["filing_date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            continue
        if fdate >= cutoff:
            hits.append(f)
    return hits


def _twilio_client():
    """Return (client, from_number, to_number) if Twilio is set up, else None.

    Reads TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER and
    ALERT_TO_NUMBER from the environment. Secrets never live in code.
    """
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_num = os.environ.get("TWILIO_FROM_NUMBER")
    to_num = os.environ.get("ALERT_TO_NUMBER")
    if not all([sid, token, from_num, to_num]):
        return None
    try:
        from twilio.rest import Client
        return Client(sid, token), from_num, to_num
    except Exception:
        return None


def send_sms_alert(message: str, to_number: Optional[str] = None) -> bool:
    """Text the message through Twilio if it is installed and configured.

    Returns True if sent. It never raises, because a failed alert should not take
    the monitor down with it.
    """
    client = _twilio_client()
    if client is None:
        print(f"[edgar] (sms not sent, Twilio not configured) {message}")
        return False
    c, from_num, to_num = client
    try:
        c.messages.create(body=message, from_=from_num, to=to_number or to_num)
        return True
    except Exception as exc:
        print(f"[edgar] Twilio SMS failed: {exc}")
        return False


def make_phone_call(message: str, to_number: Optional[str] = None) -> bool:
    """Place a Twilio voice call that reads the message aloud.

    For a filing you want to hear about before the open, a ringing phone beats a
    text you might miss. Uses inline TwiML, so there is no webhook to host.
    Returns True if the call was placed. Never raises.
    """
    client = _twilio_client()
    if client is None:
        print(f"[edgar] (call not placed, Twilio not configured) {message}")
        return False
    c, from_num, to_num = client
    try:
        c.calls.create(twiml=f"<Response><Say>{message}</Say></Response>",
                       from_=from_num, to=to_number or to_num)
        return True
    except Exception as exc:
        print(f"[edgar] Twilio call failed: {exc}")
        return False


def dispatch_alert(message: str, method: str = "sms") -> None:
    """Send an alert by text, phone call, or both, depending on `method`."""
    if method in ("sms", "both"):
        send_sms_alert(message)
    if method in ("call", "both"):
        make_phone_call(message)


def run_once(cfg: EdgarMonitorConfig,
             seen: Optional[Set[str]] = None) -> Dict[str, List[dict]]:
    """One polling cycle: return {ticker: [new dilution filings]}.

    ``seen`` (a set of accession numbers) is taken and mutated by the caller, so
    the function itself holds no global state. The same call works in a loop or as
    a one-shot.
    """
    seen = seen if seen is not None else set()
    cik_map = load_ticker_cik_map(cfg.user_agent)
    results: Dict[str, List[dict]] = {}
    for ticker in cfg.tickers:
        cik = resolve_cik(ticker, cik_map)
        if not cik:
            continue
        filings = fetch_recent_filings(cik, cfg.user_agent)
        hits = detect_dilution_filings(filings, cfg.forms, cfg.lookback_days)
        fresh = [h for h in hits if h["accession"] not in seen]
        for h in fresh:
            seen.add(h["accession"])
            dispatch_alert(f"[{ticker}] {h['form']} filed {h['filing_date']}: {h['url']}",
                           cfg.alert_method)
        if fresh:
            results[ticker] = fresh
    return results


def run_monitor(cfg: EdgarMonitorConfig) -> None:
    """Poll forever (Ctrl-C to stop). Stateful 'seen' set lives only in this loop."""
    print(f"[edgar] monitoring {', '.join(cfg.tickers)} for {', '.join(cfg.forms)} "
          f"every {cfg.poll_seconds}s")
    seen: Set[str] = set()
    while True:
        new = run_once(cfg, seen)
        if new:
            for ticker, items in new.items():
                print(f"[edgar] {ticker}: {len(items)} new dilution filing(s)")
        else:
            print(f"[edgar] {datetime.now().strftime('%H:%M:%S')} no new filings")
        time.sleep(cfg.poll_seconds)


def _main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Multi-ticker SEC EDGAR dilution monitor")
    p.add_argument("--tickers", nargs="+", default=["SPCE"])
    p.add_argument("--lookback-days", type=int, default=7)
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = p.parse_args()
    cfg = EdgarMonitorConfig(tickers=[t.upper() for t in args.tickers],
                             lookback_days=args.lookback_days)
    if args.once:
        res = run_once(cfg)
        if not res:
            print("[edgar] no recent dilution filings found")
        for ticker, items in res.items():
            for h in items:
                print(f"[edgar] {ticker}: {h['form']} {h['filing_date']} {h['url']}")
        return 0
    run_monitor(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
