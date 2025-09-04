from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu


def _pvalue_2x2(a_succ, a_fail, b_succ, b_fail):
    table = np.array([[a_succ, a_fail],
                      [b_succ, b_fail]], dtype=float)
    if (table < 5).any():
        try:
            _, p = fisher_exact(table)
            return float(p)
        except Exception:
            return float("nan")
    try:
        chi2, p, _, _ = chi2_contingency(table, correction=False)
        return float(p)
    except Exception:
        return float("nan")

def segment_lifts(df: pd.DataFrame, seg_cols: List[str], min_total: int = 500) -> Dict[str, Any]:
    """
    For each segment column and value (with nA+nB >= min_total):
      - conversion rate/lift + p-value
      - ARPU/lift + MWU p-value
    """
    rows = []
    for col in seg_cols:
        if col not in df.columns:
            continue
        for val, g in df.groupby(col):
            nA = int((g["variant"]=="A").sum())
            nB = int((g["variant"]=="B").sum())
            if nA + nB < int(min_total):
                continue

            a_succ = int(((g["variant"]=="A") & (g["converted"]==1)).sum())
            b_succ = int(((g["variant"]=="B") & (g["converted"]==1)).sum())
            pA = a_succ / nA if nA else np.nan
            pB = b_succ / nB if nB else np.nan
            conv_lift = (pB - pA) if (np.isfinite(pA) and np.isfinite(pB)) else np.nan
            p_conv = _pvalue_2x2(a_succ, nA - a_succ, b_succ, nB - b_succ)

            a_rev = g.loc[g["variant"]=="A", "revenue"].astype(float).values
            b_rev = g.loc[g["variant"]=="B", "revenue"].astype(float).values
            arpuA = float(np.mean(a_rev)) if a_rev.size else np.nan
            arpuB = float(np.mean(b_rev)) if b_rev.size else np.nan
            arpu_lift = (arpuB - arpuA) if (np.isfinite(arpuA) and np.isfinite(arpuB)) else np.nan
            try:
                _, p_mwu = mannwhitneyu(a_rev, b_rev, alternative="two-sided")
                p_mwu = float(p_mwu)
            except Exception:
                p_mwu = float("nan")

            rows.append({
                "segment": col,
                "value": str(val),
                "nA": nA, "nB": nB,
                "conv_A": pA, "conv_B": pB, "conv_lift_B_minus_A": conv_lift, "conv_p": p_conv,
                "arpu_A": arpuA, "arpu_B": arpuB, "arpu_lift_B_minus_A": arpu_lift, "arpu_p_mwu": p_mwu
            })

    rows = sorted(rows, key=lambda r: (r["nA"] + r["nB"]), reverse=True)
    return {"rows": rows, "min_total": int(min_total)}
