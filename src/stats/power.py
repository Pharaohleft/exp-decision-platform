import math
from typing import Optional

from scipy.stats import norm


def _clamp01(x: float) -> float:
    return min(max(float(x), 1e-9), 1 - 1e-9)

def required_n_per_arm(p0: float, mde_abs: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    Equal allocation, two-sided test for difference in proportions.
    p0 = baseline rate (e.g., 0.10), mde_abs = absolute lift (e.g., 0.01 for +1pp).
    Returns n PER ARM.
    """
    p0 = _clamp01(p0)
    mde_abs = float(mde_abs)
    if mde_abs <= 0:
        return float("nan")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    # Approx with p̄ ≈ p0 + mde/2
    p1 = _clamp01(p0 + mde_abs)
    pbar = (p0 + p1) / 2.0
    n = ((z_alpha + z_beta) ** 2) * (2 * pbar * (1 - pbar)) / (mde_abs ** 2)
    return float(math.ceil(n))

def mde_abs_for_n(p0: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    Given n PER ARM, return approximate absolute MDE (proportion lift).
    Uses fixed-point refinement of p̄.
    """
    p0 = _clamp01(p0)
    n = max(1, int(n_per_arm))
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    mde = (z_alpha + z_beta) * math.sqrt(2 * p0 * (1 - p0) / n)
    for _ in range(3):
        p1 = _clamp01(p0 + mde)
        pbar = (p0 + p1) / 2.0
        mde = (z_alpha + z_beta) * math.sqrt(2 * pbar * (1 - pbar) / n)
    return float(mde)

def power_for_n(p0: float, effect_abs: float, n_per_arm: int, alpha: float = 0.05) -> float:
    """
    Approximate 2-sided power given effect (absolute lift), n per arm.
    """
    p0 = _clamp01(p0)
    n = max(1, int(n_per_arm))
    p1 = _clamp01(p0 + effect_abs)
    se_alt = math.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
    if se_alt <= 0:
        return 0.0
    delta = abs(p1 - p0)
    z_alpha = norm.ppf(1 - alpha / 2)
    lam = delta / se_alt
    # Two-sided power approximation
    pw = 1 - norm.cdf(z_alpha - lam) + norm.cdf(-z_alpha - lam)
    return float(max(0.0, min(1.0, pw)))

def z_obs_two_prop(xA: int, nA: int, xB: int, nB: int) -> float:
    """
    Observed z for difference in proportions (two-sample z with pooled SE).
    """
    nA, nB = int(nA), int(nB)
    if min(nA, nB) <= 0:
        return float("nan")
    pA = xA / nA
    pB = xB / nB
    p_pool = (xA + xB) / (nA + nB)
    se0 = math.sqrt(p_pool * (1 - p_pool) * (1 / nA + 1 / nB))
    if se0 == 0:
        return float("nan")
    return float((pB - pA) / se0)

def pocock_boundary_z(looks: int, alpha: float = 0.05) -> float:
    """
    Approximate constant Pocock critical z for 2–6 looks (two-sided α≈0.05).
    ~2.41 works well across common L. If L<=1, fall back to fixed 1.96.
    """
    if looks <= 1:
        return float(norm.ppf(1 - alpha / 2))
    return 2.41
import numpy as np
from scipy.stats import norm


def obf_boundary_z(info_frac: float, alpha: float = 0.05) -> float:
    """
    Two-sided O'Brien–Fleming boundary at information fraction t in (0,1].
    Uses the classic spending approximation:
        alpha(t) = 2 - 2 * Phi( z_{alpha/2} / sqrt(t) )
    Then converts to a one-look critical value at t:
        z*(t) = Phi^{-1}(1 - alpha(t)/2)
    - Very conservative early, equals z_{alpha/2} at t=1.
    """
    t = float(max(1e-6, min(1.0, info_frac)))
    z_a2 = norm.ppf(1.0 - alpha / 2.0)
    alpha_t = 2.0 - 2.0 * norm.cdf(z_a2 / np.sqrt(t))
    z_star = norm.ppf(1.0 - alpha_t / 2.0)
    return float(z_star)
