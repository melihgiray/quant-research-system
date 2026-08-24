"""Bet sizing from classifier probabilities (Lopez de Prado, ch. 10).

A meta-model outputs a probability that its side is right; this turns that
probability into a bet size in [-1, 1]. The probability is first converted to a
test statistic against a coin flip, ``z = (p - 0.5) / sqrt(p (1 - p))``, then
mapped through the standard-normal CDF to ``2 * Phi(z) - 1``, so a coin-flip
probability sizes to zero and certainty sizes to a full bet. Discretising the
sizes into steps stops tiny probability wiggles from churning the book.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bet_size(prob, side=1.0):
    """Bet size in [-1, 1] from the probability the ``side`` is correct.

    ``prob`` is the model's probability for the predicted class; ``side`` is the
    direction (+1 long, -1 short). A probability of 0.5 sizes to zero.
    """
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1.0 - p))
    return (2.0 * norm.cdf(z) - 1.0) * np.asarray(side, dtype=float)


def discretize_bets(size, step=0.05):
    """Round bet sizes to the nearest ``step`` to avoid churning on small moves."""
    if step <= 0:
        raise ValueError("step must be positive")
    return np.round(np.asarray(size, dtype=float) / step) * step
