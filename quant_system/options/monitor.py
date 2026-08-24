"""Fail-loud health checks for a repriced option surface.

A scheduled repricing job is only useful if it refuses to trust bad data. Vendor
chains go stale, empty out, or fail to invert, and a surface built from them
looks fine until you rely on it. ``repricing_health`` turns a built surface into
a pass/fail verdict with reasons, so the job can exit non-zero and shout rather
than quietly publish garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RepricingHealth:
    """The verdict on one repriced surface."""
    ok: bool
    n_points: int
    iv_failures: int
    iv_failure_rate: float
    tradeable_arb: int
    issues: List[str] = field(default_factory=list)

    def summary(self) -> List[str]:
        head = "HEALTHY" if self.ok else "UNHEALTHY"
        lines = [f"repricing health: {head}",
                 f"  invertible quotes : {self.n_points}",
                 f"  iv-failure rate   : {self.iv_failure_rate:.0%} ({self.iv_failures} failed)",
                 f"  tradeable arb     : {self.tradeable_arb}"]
        for issue in self.issues:
            lines.append(f"  ! {issue}")
        return lines


def repricing_health(surface, max_iv_failure_rate: float = 0.6,
                     max_tradeable_arb: int = 0) -> RepricingHealth:
    """Grade a built ``VolSurface`` for whether it is safe to rely on.

    Fails on an empty surface (no invertible quotes), an implied-vol failure rate
    above ``max_iv_failure_rate`` (a chain that mostly would not invert is
    suspect), or more than ``max_tradeable_arb`` arbitrage violations that exceed
    the local bid-ask spread (the ones that usually mean genuinely stale data,
    as opposed to mid-price artefacts).
    """
    n_points = int(len(surface.points))
    n_failures = int(sum(surface.iv_failures.values())) if surface.iv_failures else 0
    total = n_points + n_failures
    failure_rate = (n_failures / total) if total else 1.0
    tradeable = len(surface.arbitrage.tradeable)

    issues: List[str] = []
    if n_points == 0:
        issues.append("no invertible quotes: empty chain or every implied vol failed")
    if failure_rate > max_iv_failure_rate:
        issues.append(f"implied-vol failure rate {failure_rate:.0%} exceeds "
                      f"{max_iv_failure_rate:.0%}")
    if tradeable > max_tradeable_arb:
        issues.append(f"{tradeable} arbitrage violation(s) beyond the bid-ask spread "
                      "(likely stale data)")

    return RepricingHealth(ok=not issues, n_points=n_points, iv_failures=n_failures,
                           iv_failure_rate=failure_rate, tradeable_arb=tradeable, issues=issues)
