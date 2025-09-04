from typing import Callable, Tuple

import numpy as np
import pandas as pd


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    group_col: str,
    stat_fn: Callable[[pd.DataFrame], float],
    B: int = 800,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Unstratified clustered bootstrap (may produce NaNs on tiny datasets if an arm is missing).
    """
    rng = np.random.default_rng(seed)
    groups = df[group_col].dropna().unique()
    if len(groups) == 0:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(B):
        samp_ids = rng.choice(groups, size=len(groups), replace=True)
        # replicate clusters according to draws
        parts = [df[df[group_col] == uid] for uid in samp_ids]
        samp = pd.concat(parts, ignore_index=True) if parts else df.head(0)
        stats.append(stat_fn(samp))
    lo, hi = np.nanpercentile(stats, [alpha/2*100, (1 - alpha/2)*100])
    return float(lo), float(hi)

def stratified_cluster_bootstrap_ci(
    df: pd.DataFrame,
    group_col: str,
    variant_col: str,
    stat_fn: Callable[[pd.DataFrame], float],
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 123,
) -> Tuple[float, float]:
    """
    Stratified clustered bootstrap by arm:
      - sample user clusters WITHIN A and WITHIN B (with replacement)
      - preserves arm sizes so each sample always has both A and B
    """
    rng = np.random.default_rng(seed)

    A_ids = df.loc[df[variant_col] == "A", group_col].dropna().unique()
    B_ids = df.loc[df[variant_col] == "B", group_col].dropna().unique()
    nA, nB = len(A_ids), len(B_ids)
    if nA == 0 or nB == 0:
        return (float("nan"), float("nan"))

    stats = []
    for _ in range(B):
        sampA = rng.choice(A_ids, size=nA, replace=True)
        sampB = rng.choice(B_ids, size=nB, replace=True)
        # replicate clusters according to draws (keep duplicates)
        parts = [df[df[group_col] == uid] for uid in sampA]
        parts += [df[df[group_col] == uid] for uid in sampB]
        samp = pd.concat(parts, ignore_index=True)
        stats.append(stat_fn(samp))

    lo, hi = np.nanpercentile(stats, [alpha/2*100, (1 - alpha/2)*100])
    return float(lo), float(hi)
