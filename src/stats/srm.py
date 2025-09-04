from typing import Dict

from scipy.stats import chisquare


def srm_from_counts(nA: int, nB: int, ratioA: float = 0.5) -> Dict[str, float | int | bool]:
    """
    Sample Ratio Mismatch (SRM) check.
    Compares observed counts [nA, nB] to expected counts under a planned split (default 50/50).
    Returns chi2, p-value, and a 'flag' that is True when p < 0.01 (red flag).
    """
    nA = int(nA); nB = int(nB)
    total = nA + nB
    if total <= 0:
        return {
            "nA": nA, "nB": nB,
            "expected_A": 0.0, "expected_B": 0.0,
            "chi2": float("nan"), "p": float("nan"),
            "flag": True, "note": "No users."
        }
    ratioA = float(max(0.0, min(1.0, ratioA)))
    expA = total * ratioA
    expB = total - expA
    chi, p = chisquare([nA, nB], f_exp=[expA, expB])
    return {
        "nA": nA, "nB": nB,
        "expected_A": float(expA), "expected_B": float(expB),
        "chi2": float(chi), "p": float(p),
        "flag": bool(p < 0.01)  # stricter than usual to catch assignment bugs
    }
