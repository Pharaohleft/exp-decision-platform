from typing import Tuple

import numpy as np
import pandas as pd


def _stratified_bootstrap_lift_fast(user_df: pd.DataFrame, value_col: str, B: int = 800, alpha: float = 0.05, seed: int = 42) -> Tuple[float,float]:
    """
    Fast stratified clustered bootstrap by user:
    - Pre-aggregate to one row per (user_id, variant) BEFORE calling this.
    - Draw multinomial counts within A and within B, then compute weighted means.
    """
    A = user_df[user_df["variant"]=="A"][value_col].to_numpy(dtype=float)
    Bv= user_df[user_df["variant"]=="B"][value_col].to_numpy(dtype=float)
    nA, nB = A.size, Bv.size
    if nA==0 or nB==0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    pA = np.full(nA, 1.0/nA)
    pB = np.full(nB, 1.0/nB)
    lifts = np.empty(B, dtype=float)
    for i in range(B):
        cA = rng.multinomial(nA, pA)   # counts sum to nA
        cB = rng.multinomial(nB, pB)   # counts sum to nB
        meanA = (cA @ A) / nA
        meanB = (cB @ Bv) / nB
        lifts[i] = meanB - meanA
    lo, hi = np.quantile(lifts, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def fast_conv_ci(user_df: pd.DataFrame, **kw) -> Tuple[float,float]:
    return _stratified_bootstrap_lift_fast(user_df, "converted", **kw)

def fast_arpu_ci(user_df: pd.DataFrame, **kw) -> Tuple[float,float]:
    return _stratified_bootstrap_lift_fast(user_df, "revenue", **kw)
