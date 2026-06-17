"""Small statistics helpers for evaluation (no scipy dependency).

Win-rates in HUNL are aggregated over a handful of seed means (typically n=3-5),
so a Normal 1.96 multiplier under-covers — the correct multiplier is Student-t with
df = n-1. These helpers provide the t critical values and a consistent aggregator
used across the eval scripts and the duplicate-hand evaluator.
"""

import numpy as np

# Two-sided 95% Student-t critical values by degrees of freedom (df = n - 1).
# For df > 30 the t distribution is within ~3% of Normal, so we fall back to 1.96.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def t95_multiplier(n: int) -> float:
    """Two-sided 95% Student-t multiplier for a sample of size ``n`` (df = n - 1).

    Returns 0.0 for n <= 1 (no spread estimable) and the Normal 1.96 for df > 30.
    """
    df = int(n) - 1
    if df <= 0:
        return 0.0
    if df > 30:
        return 1.96
    return _T95[df]


def aggregate_ci95(values):
    """Mean and 95% Student-t CI half-width over a small sample of seed means.

    Returns a dict with mean, std (ddof=1), stderr, ci95 (half-width), n, and the
    ci_method actually used ("student_t" or "normal").
    """
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean()) if n else 0.0
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    stderr = float(std / np.sqrt(n)) if n > 1 else 0.0
    ci95 = float(t95_multiplier(n) * stderr)
    return {
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "ci95": ci95,
        "n": n,
        "ci_method": "normal" if (n - 1) > 30 else "student_t",
    }
