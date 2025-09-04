from typing import Tuple

import numpy as np


def mean_perm_test(a: np.ndarray, b: np.ndarray, B: int = 3000, seed: int = 123) -> Tuple[float, float]:
    """
    Two-sample permutation test for difference in means (B - A).
    Returns (obs_diff, two_sided_p).
    NaNs are ignored.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    nA, nB = a.size, b.size
    if nA == 0 or nB == 0:
        return (float("nan"), float("nan"))
    obs = float(b.mean() - a.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(B)):
        rng.shuffle(pooled)
        A_ = pooled[:nA]
        B_ = pooled[nA:]
        diff = B_.mean() - A_.mean()
        if abs(diff) >= abs(obs):
            count += 1
    p = (count + 1) / (B + 1)  # add-one smoothing
    return obs, float(p)
