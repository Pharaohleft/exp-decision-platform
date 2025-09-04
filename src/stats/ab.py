from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha/2)
    phat = successes / n
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    half = z * np.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(lo), float(hi)

def chi_square_2x2(a_succ:int, a_fail:int, b_succ:int, b_fail:int):
    table = np.array([[a_succ, a_fail],
                      [b_succ, b_fail]], dtype=float)
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p), int(dof)

def conversion_summary(df: pd.DataFrame):
    # counts
    nA = int((df["variant"]=="A").sum())
    nB = int((df["variant"]=="B").sum())
    a_succ = int(((df["variant"]=="A") & (df["converted"]==1)).sum())
    b_succ = int(((df["variant"]=="B") & (df["converted"]==1)).sum())

    # rates
    pA = a_succ / nA if nA else float("nan")
    pB = b_succ / nB if nB else float("nan")
    lift = (pB - pA) if (pA==pA and pB==pB) else float("nan")

    # CIs for each arm
    ciA = wilson_ci(a_succ, nA) if nA else (float("nan"), float("nan"))
    ciB = wilson_ci(b_succ, nB) if nB else (float("nan"), float("nan"))

    # chi-square p-value
    chi2, pval, _ = chi_square_2x2(a_succ, nA - a_succ, b_succ, nB - b_succ)

    # simple recommendation
    recommend = "Not enough evidence"
    if (lift is not None) and (lift > 0) and (pval <= 0.05):
        recommend = "Looks like a win (statistically convincing)"
    elif (lift is not None) and (lift < 0) and (pval <= 0.05):
        recommend = "Looks worse (statistically convincing)"

    return {
        "nA": nA, "nB": nB,
        "conv_A": pA, "conv_B": pB,
        "conv_CI_A": ciA, "conv_CI_B": ciB,
        "lift_B_minus_A": lift,
        "chi2_p": pval,
        "recommendation": recommend
    }
import numpy as np
# -------- ARPU + CUPED --------
from scipy import stats


def cuped_adjust(y: np.ndarray, x: np.ndarray):
    """
    CUPED adjustment: Y_adj = Y - theta * (X - mean(X)),
    where theta = cov(Y,X)/var(X). Returns (Y_adj, theta).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    vx = np.var(x, ddof=1)
    if vx <= 0 or len(y) != len(x):
        return y.copy(), 0.0
    theta = np.cov(y, x, ddof=1)[0, 1] / vx
    y_adj = y - theta * (x - np.mean(x))
    return y_adj, float(theta)

def mann_whitney_p(a: np.ndarray, b: np.ndarray):
    """Two-sided Mann–Whitney U (robust for skewed revenue)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")

def revenue_summary(df: pd.DataFrame):
    """
    Returns revenue (ARPU) stats:
      - Means by arm
      - Lift (B-A)
      - Mann–Whitney U p-value on raw revenue
      - CUPED theta and variance reduction %
      - CUPED-adjusted mean-lift (B-A)
    """
    a = df.loc[df["variant"] == "A", "revenue"].astype(float).values
    b = df.loc[df["variant"] == "B", "revenue"].astype(float).values
    pre_a = df.loc[df["variant"] == "A", "pre_metric"].astype(float).values
    pre_b = df.loc[df["variant"] == "B", "pre_metric"].astype(float).values

    arpuA = float(np.mean(a)) if len(a) else float("nan")
    arpuB = float(np.mean(b)) if len(b) else float("nan")
    lift_raw = arpuB - arpuA if (arpuA == arpuA and arpuB == arpuB) else float("nan")

    # Robust test on raw revenue
    p_mwu = mann_whitney_p(a, b)

    # CUPED adjustment using pre_metric
    y_adj_A, thetaA = cuped_adjust(a, pre_a) if len(a) and len(pre_a) else (a, 0.0)
    y_adj_B, thetaB = cuped_adjust(b, pre_b) if len(b) and len(pre_b) else (b, 0.0)

    # Use a single theta report (average), and compute VR% against each arm's raw variance
    theta = float(np.nanmean([thetaA, thetaB]))
    vr_A = 1.0 - (np.var(y_adj_A, ddof=1) / np.var(a, ddof=1)) if len(a) > 1 and np.var(a, ddof=1) > 0 else 0.0
    vr_B = 1.0 - (np.var(y_adj_B, ddof=1) / np.var(b, ddof=1)) if len(b) > 1 and np.var(b, ddof=1) > 0 else 0.0
    vr_pct = float(100 * max(0.0, np.nanmean([vr_A, vr_B])))

    arpuA_cuped = float(np.mean(y_adj_A)) if len(y_adj_A) else float("nan")
    arpuB_cuped = float(np.mean(y_adj_B)) if len(y_adj_B) else float("nan")
    lift_cuped = arpuB_cuped - arpuA_cuped if (arpuA_cuped == arpuA_cuped and arpuB_cuped == arpuB_cuped) else float("nan")

    # Simple green/grey flag just for ARPU dimension
    arpu_flag = "green" if (lift_raw > 0 and p_mwu <= 0.05) else "grey"

    return {
        "arpu_A": arpuA,
        "arpu_B": arpuB,
        "lift_arpu_B_minus_A": lift_raw,
        "mann_whitney_p": p_mwu,
        "cuped_theta": theta,
        "cuped_variance_reduction_pct": vr_pct,
        "arpu_cuped_A": arpuA_cuped,
        "arpu_cuped_B": arpuB_cuped,
        "lift_arpu_cuped_B_minus_A": lift_cuped,
        "arpu_flag": arpu_flag,
        "notes": "MWU p-value is on RAW revenue (robust). CUPED is reported for mean lift & VR%."
    }

