from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

# step name synonyms to order a typical funnel if those columns exist
_ORDER_HINTS = [
    ("viewed","view","impression","seen"),
    ("clicked","click","tap"),
    ("add_to_cart","added_to_cart","atc"),
    ("checkout","checked_out"),
    ("purchase","purchased","ordered","converted"),  # final
]

def _is_binary_series(s: pd.Series) -> bool:
    try:
        vals = pd.unique(pd.to_numeric(s, errors="coerce").fillna(-1))
        return set(vals).issubset({0,1})
    except Exception:
        return False

def _order_steps(cols: List[str]) -> List[str]:
    assigned = set()
    ordered = []
    # use hints first
    for group in _ORDER_HINTS:
        for name in group:
            for c in cols:
                if c.lower() == name and c not in assigned:
                    ordered.append(c); assigned.add(c); break
            if ordered and ordered[-1].lower() == name:
                break
    # add any remaining binary cols alphabetically
    for c in sorted(cols):
        if c not in assigned:
            ordered.append(c)
    return ordered

def funnel_summary(df: pd.DataFrame) -> Dict[str, Any]:
    # auto-detect candidate binary columns (0/1), exclude known non-funnel
    exclude = {"variant","revenue","pre_metric","user_id"}
    bin_cols = [c for c in df.columns if c not in exclude and _is_binary_series(df[c])]
    if not bin_cols:
        # ensure at least "converted" exists
        if "converted" in df.columns:
            bin_cols = ["converted"]
        else:
            return {"steps": [], "note": "No binary funnel columns found."}

    steps = _order_steps(bin_cols)

    out_rows = []
    for step in steps:
        a = df.loc[df["variant"]=="A", step].astype(float)
        b = df.loc[df["variant"]=="B", step].astype(float)
        nA, nB = a.size, b.size
        a_succ, b_succ = int(a.sum()), int(b.sum())
        pA, pB = (a.mean(), b.mean())
        # 2x2 p-value
        table = np.array([[a_succ, nA - a_succ],[b_succ, nB - b_succ]], dtype=float)
        if (table < 5).any():
            try:
                _, p = fisher_exact(table)
            except Exception:
                p = np.nan
        else:
            try:
                _, p, _, _ = chi2_contingency(table, correction=False)
            except Exception:
                p = np.nan

        out_rows.append({
            "step": step,
            "rate_A": float(pA), "rate_B": float(pB),
            "lift_B_minus_A": float(pB - pA),
            "p_value": float(p) if np.isfinite(p) else float("nan"),
            "nA": int(nA), "nB": int(nB)
        })

    return {"steps": out_rows}
