"""Static no-arbitrage checks on a volatility surface.

A fitted vol surface can look beautiful and still imply negative probabilities.
Two classic conditions catch most of it, and both are checked here rather than
being smoothed away, because a violation is information: it usually means a
stale quote, a bad spot, or a mis-specified dividend, and papering over it hides
the data problem instead of fixing it.

**Butterfly (convexity in strike).** Call prices must be convex in strike at a
fixed expiry. The butterfly spread (long one K1, short two K2, long one K3) is a
portfolio that can never pay out less than zero, so it cannot cost less than
zero. If it does, the surface implies a negative risk-neutral density over that
strike range. Prices must also be non-increasing in strike, since the right to
buy at a lower price is worth more.

**Calendar (monotonic total variance).** Total variance ``w = sigma^2 * T`` must
be non-decreasing in expiry at a fixed log-moneyness. Uncertainty accumulates:
an option cannot become less uncertain about where the stock lands by giving it
more time. A dip in total variance means the longer-dated option is cheap enough
to arbitrage against the shorter one.

Every violation is returned with its location and magnitude, logged at warning
level, and counted. Nothing is silently repaired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BUTTERFLY = "butterfly"
CALENDAR = "calendar"
MONOTONICITY = "strike_monotonicity"


@dataclass(frozen=True)
class ArbitrageViolation:
    """One violated no-arbitrage condition, located and quantified.

    ``within_spread`` is the difference between an interesting violation and a
    boring one. Surfaces are built from mid prices, but you cannot trade at mid.
    A butterfly that is 0.5 cents negative on a market that is 23 cents wide is
    an artifact of taking mids, not an opportunity: capturing it means crossing
    a spread far larger than the edge. A violation that exceeds the local spread
    is the one worth investigating, and usually means genuinely stale data.
    """

    kind: str
    severity: float           # how far into violation, in price or variance units
    detail: str
    expiry: object = None
    strike: float = float("nan")
    log_moneyness: float = float("nan")
    within_spread: Optional[bool] = None

    def __str__(self) -> str:
        tag = ""
        if self.within_spread is True:
            tag = " [inside bid-ask, not tradeable]"
        elif self.within_spread is False:
            tag = " [EXCEEDS bid-ask]"
        return f"[{self.kind}] {self.detail} (severity {self.severity:.3e}){tag}"


@dataclass
class ArbitrageReport:
    """Violations found, plus how much was checked to find them."""

    violations: List[ArbitrageViolation] = field(default_factory=list)
    n_butterfly_checks: int = 0
    n_calendar_checks: int = 0

    @property
    def clean(self) -> bool:
        return not self.violations

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in self.violations:
            out[v.kind] = out.get(v.kind, 0) + 1
        return out

    def worst(self, kind: str = None) -> ArbitrageViolation:
        pool = [v for v in self.violations if kind is None or v.kind == kind]
        return max(pool, key=lambda v: v.severity) if pool else None

    @property
    def tradeable(self) -> List[ArbitrageViolation]:
        """Violations larger than the local bid-ask, i.e. not a mid-price artifact."""
        return [v for v in self.violations if v.within_spread is False]

    def summary(self, max_detail: int = 5) -> List[str]:
        total = self.n_butterfly_checks + self.n_calendar_checks
        lines = [f"NO-ARBITRAGE CHECKS ({total} conditions tested: "
                 f"{self.n_butterfly_checks} butterfly, {self.n_calendar_checks} calendar)"]
        if self.clean:
            lines.append(" no violations found")
            return lines
        for kind, n in sorted(self.counts().items()):
            lines.append(f" {kind}: {n} violation(s)")
        graded = [v for v in self.violations if v.within_spread is not None]
        if graded:
            big = sum(1 for v in graded if v.within_spread is False)
            lines.append(f" {big} of {len(graded)} exceed the local bid-ask spread; "
                         f"the rest are mid-price artifacts")
        for v in sorted(self.violations, key=lambda x: -x.severity)[:max_detail]:
            lines.append(f"   {v}")
        if len(self.violations) > max_detail:
            lines.append(f"   ... and {len(self.violations) - max_detail} more")
        return lines


def check_butterfly(points: pd.DataFrame, tolerance: float = 1e-6) -> List[ArbitrageViolation]:
    """Convexity and monotonicity of call prices in strike, per expiry.

    Parameters
    ----------
    points : pd.DataFrame
        Needs ``expiry``, ``strike`` and ``call_price`` columns. The call price
        is the Black-Scholes price at that point's implied vol, so this tests
        the surface rather than the raw quotes.
    tolerance : float
        Absolute slack in price units before something counts as a violation.

    Returns
    -------
    list[ArbitrageViolation]
    """
    violations: List[ArbitrageViolation] = []

    has_spread = "spread" in points.columns

    for expiry, group in points.groupby("expiry", sort=True):
        slice_ = group.sort_values("strike")
        strikes = slice_["strike"].to_numpy(dtype=float)
        prices = slice_["call_price"].to_numpy(dtype=float)
        spreads = (slice_["spread"].to_numpy(dtype=float) if has_spread
                   else np.full(len(strikes), np.nan))
        if len(strikes) < 2:
            continue

        # Calls must be non-increasing in strike.
        for i in range(len(strikes) - 1):
            gap = prices[i + 1] - prices[i]
            if gap > tolerance:
                violations.append(ArbitrageViolation(
                    kind=MONOTONICITY, severity=float(gap), expiry=expiry,
                    strike=float(strikes[i + 1]),
                    detail=(f"call price rises with strike at {expiry:%Y-%m-%d}: "
                            f"C({strikes[i]:.1f})={prices[i]:.4f} < "
                            f"C({strikes[i + 1]:.1f})={prices[i + 1]:.4f}")))

        # Convexity: the butterfly built from each consecutive triplet costs >= 0.
        for i in range(len(strikes) - 2):
            k1, k2, k3 = strikes[i], strikes[i + 1], strikes[i + 2]
            if k3 <= k1:
                continue
            w1 = (k3 - k2) / (k3 - k1)
            w3 = (k2 - k1) / (k3 - k1)
            cost = w1 * prices[i] + w3 * prices[i + 2] - prices[i + 1]
            if cost < -tolerance:
                # The butterfly costs three legs, so the relevant hurdle is the
                # widest market among them.
                local_spread = np.nanmax(spreads[i:i + 3]) if has_spread else np.nan
                inside = (bool(-cost < local_spread)
                          if np.isfinite(local_spread) else None)
                violations.append(ArbitrageViolation(
                    kind=BUTTERFLY, severity=float(-cost), expiry=expiry,
                    strike=float(k2), within_spread=inside,
                    detail=(f"negative butterfly at {expiry:%Y-%m-%d} "
                            f"K={k1:.1f}/{k2:.1f}/{k3:.1f}: cost {cost:.6f}")))

    return violations


def check_calendar(points: pd.DataFrame, tolerance: float = 1e-6) -> List[ArbitrageViolation]:
    """Total variance must not fall as expiry lengthens, at matched moneyness.

    Compares each adjacent pair of expiries on the log-moneyness range they
    share, interpolating the shorter slice onto the longer one's grid.
    """
    violations: List[ArbitrageViolation] = []
    expiries = sorted(points["expiry"].unique())
    if len(expiries) < 2:
        return violations

    for near_exp, far_exp in zip(expiries[:-1], expiries[1:]):
        near = points[points["expiry"] == near_exp].sort_values("log_moneyness")
        far = points[points["expiry"] == far_exp].sort_values("log_moneyness")
        if near.empty or far.empty:
            continue

        k_near = near["log_moneyness"].to_numpy(dtype=float)
        w_near = near["total_variance"].to_numpy(dtype=float)
        k_far = far["log_moneyness"].to_numpy(dtype=float)
        w_far = far["total_variance"].to_numpy(dtype=float)

        # Only compare where the two slices actually overlap in moneyness;
        # extrapolating would manufacture violations that are not in the data.
        lo = max(k_near.min(), k_far.min())
        hi = min(k_near.max(), k_far.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue

        mask = (k_far >= lo) & (k_far <= hi)
        for k, w_long in zip(k_far[mask], w_far[mask]):
            w_short = float(np.interp(k, k_near, w_near))
            drop = w_short - w_long
            if drop > tolerance:
                violations.append(ArbitrageViolation(
                    kind=CALENDAR, severity=float(drop), expiry=far_exp,
                    log_moneyness=float(k),
                    detail=(f"total variance falls from {near_exp:%Y-%m-%d} "
                            f"({w_short:.6f}) to {far_exp:%Y-%m-%d} ({w_long:.6f}) "
                            f"at k={k:+.3f}")))

    return violations


def check_surface(points: pd.DataFrame, butterfly_tolerance: float = 1e-6,
                  calendar_tolerance: float = 1e-6) -> ArbitrageReport:
    """Run every static check and return a single report.

    Logs a warning per violation kind when anything is found, so a surface built
    inside a scheduled job leaves a trail rather than failing silently.
    """
    butterfly = check_butterfly(points, butterfly_tolerance)
    calendar = check_calendar(points, calendar_tolerance)

    n_bf = 0
    for _, group in points.groupby("expiry"):
        n = len(group)
        n_bf += max(n - 2, 0) + max(n - 1, 0)
    n_cal = _count_calendar_checks(points)

    report = ArbitrageReport(violations=butterfly + calendar,
                             n_butterfly_checks=n_bf,
                             n_calendar_checks=n_cal)

    if not report.clean:
        for kind, n in sorted(report.counts().items()):
            worst = report.worst(kind)
            logger.warning("surface arbitrage: %d %s violation(s), worst: %s",
                           n, kind, worst.detail if worst else "n/a")
    return report


def _count_calendar_checks(points: pd.DataFrame) -> int:
    """How many calendar comparisons were possible (for an honest denominator)."""
    expiries = sorted(points["expiry"].unique())
    total = 0
    for near_exp, far_exp in zip(expiries[:-1], expiries[1:]):
        near = points[points["expiry"] == near_exp]
        far = points[points["expiry"] == far_exp]
        if near.empty or far.empty:
            continue
        lo = max(near["log_moneyness"].min(), far["log_moneyness"].min())
        hi = min(near["log_moneyness"].max(), far["log_moneyness"].max())
        if hi > lo:
            total += int(((far["log_moneyness"] >= lo)
                          & (far["log_moneyness"] <= hi)).sum())
    return total
