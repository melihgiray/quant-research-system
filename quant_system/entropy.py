"""Entropy of a discretised return series (Lopez de Prado, ch. 18).

Encoding returns as up/down symbols and measuring the entropy of the resulting
message asks how predictable the sequence is: a fair coin has maximal entropy
(nothing to predict), while a sequence with structure has less. It is a
model-free complement to autocorrelation and the variance ratio.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def returns_to_bits(returns, threshold: float = 0.0) -> str:
    """Encode returns as a bit string: '1' for a move above ``threshold`` else '0'."""
    values = np.asarray(returns, dtype=float)
    return "".join("1" if r > threshold else "0" for r in values)


def plugin_entropy(message: str, word_length: int = 1) -> float:
    """Plug-in Shannon entropy per symbol from the empirical word distribution.

    Slides a window of ``word_length`` over the message, estimates the probability
    of each distinct word by its frequency, and returns Shannon entropy in bits
    divided by the word length. A fair random string tends to 1 bit per symbol; a
    fully predictable one tends to 0.
    """
    if word_length < 1:
        raise ValueError("word_length must be >= 1")
    if len(message) < word_length:
        return float("nan")
    words = [message[i:i + word_length] for i in range(len(message) - word_length + 1)]
    counts = Counter(words)
    total = len(words)
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / word_length)
