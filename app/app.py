# app/app.py
from __future__ import annotations

import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------------
# Hotfix for a gradio-client 1.3.0 bug:
# "TypeError: argument of type 'bool' is not iterable" when building API info.
# We defensively handle boolean JSON Schemas such as {"additionalProperties": True}
# so the UI can load.
# -----------------------------------------------------------------------------
try:
    from gradio_client import utils as _gutils  # type: ignore

    if hasattr(_gutils, "get_type") and hasattr(_gutils, "_json_schema_to_python_type"):
        _orig_get_type = _gutils.get_type
        _orig_json_to_py = _gutils._json_schema_to_python_type

        def _patched_get_type(schema):  # type: ignore
            if isinstance(schema, bool):
                return "any"
            return _orig_get_type(schema)

        def _patched_json_to_py(schema, defs=None):  # type: ignore
            if isinstance(schema, bool):
                return "Any"
            if isinstance(schema, dict):
                ap = schema.get("additionalProperties", None)
                if isinstance(ap, bool):
                    # Convert True/False into a permissive object schema so the
                    # helper doesn't try to treat a bool as a dict.
                    schema = dict(schema)
                    schema["additionalProperties"] = {}
            return _orig_json_to_py(schema, defs)

        _gutils.get_type = _patched_get_type  # type: ignore
        _gutils._json_schema_to_python_type = _patched_json_to_py  # type: ignore
except Exception:
    # If anything goes wrong here, just proceed; worst-case the original bug shows.
    pass


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
REQUIRED_COLS = {"user_id", "variant", "converted", "revenue"}
OPTIONAL_SEGMENT_COLS = ["device", "country"]
OPTIONAL_FUNNEL_COLS = ["viewed", "clicked", "add_to_cart", "converted"]


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """Coerce common truthy/falsey representations into {0,1} ints."""
    if s.dtype == bool:
        return s.astype(int)
    # strings like "true"/"false", "yes"/"no"
    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "t": 1,
        "f": 0,
        "1": 1,
        "0": 0,
    }
    try:
        return s.map(lambda x: mapping.get(str(x).strip().lower(), x)).astype(int)
    except Exception:
        return s.astype(float).fillna(0).astype(int)


def load_csv(file: gr.File | str) -> pd.DataFrame:
    if isinstance(file, str):
        path = file
    else:
        path = file.name if hasattr(file, "name") else file

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # minimal coercions
    if "converted" in df.columns:
        df["converted"] = _coerce_bool_series(df["converted"])
    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)

    # variant -> uppercase A/B
    if "variant" in df.columns:
        df["variant"] = df["variant"].astype(str).str.strip().str.upper()

    return df


# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------
def srm_check(df: pd.DataFrame) -> Dict[str, float | bool]:
    counts = df["variant"].value_counts(dropna=False)
    nA = int(counts.get("A", 0))
    nB = int(counts.get("B", 0))
    total = nA + nB if (nA + nB) > 0 else 1
    expected = total / 2.0

    chi2 = ((nA - expected) ** 2) / expected + ((nB - expected) ** 2) / expected
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return {
        "nA": nA,
        "nB": nB,
        "expected_A": expected,
        "expected_B": expected,
        "chi2": round(float(chi2), 4),
        "p": float(p),
        "flag": bool(p < 0.01),
    }


def _rate_ci(count: int, n: int, alpha=0.05) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = count / n
    lo, hi = stats.beta.ppf([alpha / 2, 1 - alpha / 2], count + 1, n - count + 1)
    return (float(lo), float(hi))


def ab_basic(df: pd.DataFrame) -> Dict[str, float]:
    gA = df[df["variant"] == "A"]
    gB = df[df["variant"] == "B"]

    nA, nB = len(gA), len(gB)
    convA = gA["converted"].mean() if nA else 0.0
    convB = gB["converted"].mean() if nB else 0.0
    arpuA = gA["revenue"].mean() if nA else 0.0
    arpuB = gB["revenue"].mean() if nB else 0.0
    return {
        "rows": float(len(df)),
        "nA": float(nA),
        "nB": float(nB),
        "conv_rate_A": float(convA),
        "conv_rate_B": float(convB),
        "lift_conv_B_minus_A": float(convB - convA),
        "arpu_A": float(arpuA),
        "arpu_B": float(arpuB),
        "arpu_lift_B_minus_A": float(arpuB - arpuA),
    }


def ab_conversion_test(df: pd.DataFrame, alpha=0.05, B=300) -> Dict:
    gA = df[df["variant"] == "A"]
    gB = df[df["variant"] == "B"]
    nA, nB = len(gA), len(gB)
    convA = gA["converted"].mean() if nA else 0.0
    convB = gB["converted"].mean() if nB else 0.0

    # Chi-square (2x2)
    table = np.array(
        [
            [gA["converted"].sum(), nA - gA["converted"].sum()],
            [gB["converted"].sum(), nB - gB["converted"].sum()],
        ]
    )
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)

    # CIs
    ciA = _rate_ci(int(gA["converted"].sum()), nA, alpha=alpha)
    ciB = _rate_ci(int(gB["converted"].sum()), nB, alpha=alpha)

    # Bootstrap for lift CI
    lift_samples = []
    if nA > 0 and nB > 0 and B > 0:
        A_arr = gA["converted"].to_numpy()
        B_arr = gB["converted"].to_numpy()
        rng = np.random.default_rng(42)
        for _ in range(B):
            mA = rng.choice(A_arr, size=nA, replace=True).mean()
            mB = rng.choice(B_arr, size=nB, replace=True).mean()
            lift_samples.append(mB - mA)
    if lift_samples:
        lo, hi = np.percentile(lift_samples, [2.5, 97.5])
        boot_ci = [float(lo), float(hi)]
    else:
        boot_ci = [float("nan"), float("nan")]

    return {
        "nA": nA,
        "nB": nB,
        "conv_A": float(convA),
        "conv_B": float(convB),
        "conv_CI_A": [float(ciA[0]), float(ciA[1])],
        "conv_CI_B": [float(ciB[0]), float(ciB[1])],
        "lift_B_minus_A": float(convB - convA),
        "chi2_p": float(p),
        "recommendation": "Not enough evidence"
        if p >= alpha
        else "Likely different",
        "bootstrap_CI_lift_95": boot_ci,
    }


def ab_revenue_test(df: pd.DataFrame, alpha=0.05, B=300) -> Dict:
    gA = df[df["variant"] == "A"]["revenue"].to_numpy()
    gB = df[df["variant"] == "B"]["revenue"].to_numpy()
    arpuA = float(np.mean(gA)) if len(gA) else 0.0
    arpuB = float(np.mean(gB)) if len(gB) else 0.0

    # Mann-Whitney (robust to skew/heavy tails)
    if len(gA) and len(gB):
        _, p = stats.mannwhitneyu(gA, gB, alternative="two-sided")
    else:
        p = 1.0

    # CUPED (simple theta via cov(pre, y)/var(pre)); here we have no pre,
    # so just report zero VR to keep shape consistent.
    cuped_theta = 0.0
    vr = 0.0

    # Bootstrap CI for ARPU lift
    lift_samples = []
    rng = np.random.default_rng(43)
    if len(gA) and len(gB) and B > 0:
        for _ in range(B):
            mA = rng.choice(gA, size=len(gA), replace=True).mean()
            mB = rng.choice(gB, size=len(gB), replace=True).mean()
            lift_samples.append(mB - mA)
    if lift_samples:
        lo, hi = np.percentile(lift_samples, [2.5, 97.5])
        boot_ci = [float(lo), float(hi)]
    else:
        boot_ci = [float("nan"), float("nan")]

    flag = "green" if p < alpha else "grey"
    return {
        "arpu_A": arpuA,
        "arpu_B": arpuB,
        "lift_arpu_B_minus_A": float(arpuB - arpuA),
        "mann_whitney_p": float(p),
        "cuped_theta": float(cuped_theta),
        "cuped_variance_reduction_pct": float(vr),
        "arpu_cuped_A": arpuA,
        "arpu_cuped_B": arpuB,
        "lift_arpu_cuped_B_minus_A": float(arpuB - arpuA),
        "arpu_flag": flag,
        "notes": "MWU p-value is on RAW revenue (robust).",
        "bootstrap_CI_lift_95": boot_ci,
    }


def segments_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["segment", "value", "nA", "nB", "conv_A", "conv_B", "conv_lift",
            "conv_p", "arpu_A", "arpu_B", "arpu_lift", "arpu_p_mwu"]
    rows: List[List] = []
    for seg in OPTIONAL_SEGMENT_COLS:
        if seg not in df.columns:
            continue
        for val, g in df.groupby(seg):
            gA = g[g["variant"] == "A"]
            gB = g[g["variant"] == "B"]
            nA, nB = len(gA), len(gB)
            cA, cB = gA["converted"].mean() if nA else 0.0, gB["converted"].mean() if nB else 0.0
            arA, arB = gA["revenue"].mean() if nA else 0.0, gB["revenue"].mean() if nB else 0.0

            # conversion p (chi2)
            if nA and nB:
                table = np.array(
                    [
                        [gA["converted"].sum(), nA - gA["converted"].sum()],
                        [gB["converted"].sum(), nB - gB["converted"].sum()],
                    ]
                )
                _, p_conv, _, _ = stats.chi2_contingency(table, correction=False)
                # revenue p (MWU)
                _, p_rev = stats.mannwhitneyu(
                    gA["revenue"].to_numpy(), gB["revenue"].to_numpy(),
                    alternative="two-sided"
                )
            else:
                p_conv, p_rev = 1.0, 1.0

            rows.append(
                [
                    seg, str(val), nA, nB,
                    float(cA), float(cB), float(cB - cA), float(p_conv),
                    float(arA), float(arB), float(arB - arA), float(p_rev),
                ]
            )
    return pd.DataFrame(rows, columns=cols)


def funnel_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Builds a simple step-wise comparison if funnel columns exist.
    steps = [c for c in OPTIONAL_FUNNEL_COLS if c in df.columns]
    cols = ["step", "rate_A", "rate_B", "lift_B_minus_A", "p_value", "nA", "nB"]
    rows: List[List] = []
    if not steps:
        return pd.DataFrame([], columns=cols)

    gA = df[df["variant"] == "A"]
    gB = df[df["variant"] == "B"]
    nA, nB = len(gA), len(gB)
    for s in steps:
        a = _coerce_bool_series(gA[s]).mean() if nA else 0.0
        b = _coerce_bool_series(gB[s]).mean() if nB else 0.0
        # two-proportion z test (approx)
        p_pool = (a * nA + b * nB) / max(nA + nB, 1)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / max(nA, 1) + 1 / max(nB, 1)))
        if se == 0:
            p = 1.0
        else:
            z = (b - a) / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append([s, float(a), float(b), float(b - a), float(p), nA, nB])

    return pd.DataFrame(rows, columns=cols)


def decision_memo(conv_p: float, rev_p: float, srm_bad: bool) -> str:
    if srm_bad:
        return "⚠️ **SRM issue detected** (uneven traffic split). Investigate before trusting results."
    if conv_p < 0.05 or rev_p < 0.05:
        return "✅ **Statistically significant** difference detected at 5% alpha."
    return "⏸️ **Hold** — not enough evidence yet; consider more sample or time."


# -----------------------------------------------------------------------------
# Gradio glue
# -----------------------------------------------------------------------------
SEG_COLS = [
    "segment", "value", "nA", "nB",
    "conv_A", "conv_B", "conv_lift", "conv_p",
    "arpu_A", "arpu_B", "arpu_lift", "arpu_p_mwu",
]
FUNNEL_COLS = ["step", "rate_A", "rate_B", "lift_B_minus_A", "p_value", "nA", "nB"]


def analyze(
    csv_file, bootstrap_B: int, include_segments: bool
) -> Tuple[str, pd.DataFrame, pd.DataFrame, str]:
    try:
        df = load_csv(csv_file)
    except Exception as e:
        return (
            f"Failed to read CSV: {e}",
            pd.DataFrame([], columns=SEG_COLS),
            pd.DataFrame([], columns=FUNNEL_COLS),
            "",
        )

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return (
            f"CSV missing required columns: {sorted(missing)}",
            pd.DataFrame([], columns=SEG_COLS),
            pd.DataFrame([], columns=FUNNEL_COLS),
            "",
        )

    srm = srm_check(df)
    basic = ab_basic(df)
    conv = ab_conversion_test(df, B=bootstrap_B)
    rev = ab_revenue_test(df, B=bootstrap_B)

    seg_df = segments_summary(df) if include_segments else pd.DataFrame([], columns=SEG_COLS)
    if include_segments and seg_df.empty:
        seg_df = pd.DataFrame([], columns=SEG_COLS)

    fun_df = funnel_summary(df)
    if fun_df.empty:
        fun_df = pd.DataFrame([], columns=FUNNEL_COLS)

    memo = decision_memo(conv_p=conv["chi2_p"], rev_p=rev["mann_whitney_p"], srm_bad=srm["flag"])

    # Build a compact report JSON and save to temp for download
    report = {
        "srm": srm,
        "basic_summary": basic,
        "conversion_test": conv,
        "revenue_test": rev,
        "segments": {"rows": seg_df.to_dict(orient="records")},
        "funnel": {"steps": fun_df.to_dict(orient="records")},
        "settings": {"bootstrap_B": bootstrap_B},
        "decision_hint": "Hold" if "Hold" in memo or "SRM" in memo else "Consider Action",
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(tmp.name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Compose a quick human-readable header
    header = []
    header.append(f"**SRM**: nA={srm['nA']}, nB={srm['nB']}, p={srm['p']:.3f} "
                  f"{'⚠️ (issue)' if srm['flag'] else ''}")
    header.append(
        f"**Conversion**: A={basic['conv_rate_A']:.3%}, "
        f"B={basic['conv_rate_B']:.3%}, "
        f"Δ={basic['lift_conv_B_minus_A']:.3%}, p={conv['chi2_p']:.3f}"
    )
    header.append(
        f"**Revenue/USR**: A={basic['arpu_A']:.3f}, "
        f"B={basic['arpu_B']:.3f}, "
        f"Δ={basic['arpu_lift_B_minus_A']:.3f}, p={rev['mann_whitney_p']:.3f}"
    )
    header.append("")
    header.append(memo)

    return (
        "\n\n".join(header),
        seg_df,
        fun_df,
        tmp.name,
    )


# --- Power calculator (simple two-proportion) --------------------------------
def power_calc(
    baseline_rate: float, abs_mde: float, alpha: float, power: float
) -> str:
    """
    Return samples per arm for a two-sided z-test on proportions.
    """
    if baseline_rate <= 0 or baseline_rate >= 1:
        return "Baseline must be in (0,1)."
    if abs_mde <= 0:
        return "MDE must be > 0."
    if alpha <= 0 or alpha >= 1 or power <= 0 or power >= 1:
        return "alpha and power must be in (0,1)."

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + abs_mde
    pbar = (p1 + p2) / 2
    qbar = 1 - pbar
    q1 = 1 - p1
    q2 = 1 - p2

    n = ((z_alpha * math.sqrt(2 * pbar * qbar) + z_beta * math.sqrt(p1 * q1 + p2 * q2)) ** 2) / (
        abs_mde ** 2
    )
    n = math.ceil(n)
    return f"≈ **{n:,} users per arm** for a two-sided test."


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Experiment Decision Helper") as demo:
        gr.Markdown("# Experiment Decision Helper\nUpload your CSV and get a one-page readout.")
        with gr.Tab("Analyze"):
            with gr.Row():
                csv_in = gr.File(label="Upload experiment CSV", file_types=[".csv"])
                boot = gr.Slider(0, 2000, value=300, step=50, label="Bootstrap resamples (for CIs)")
                seg_on = gr.Checkbox(value=True, label="Compute segment breakouts (device/country)")

            memo_md = gr.Markdown(value="—")
            seg_table = gr.Dataframe(
                headers=SEG_COLS,
                col_count=(len(SEG_COLS), "fixed"),
                row_count=(1, "dynamic"),
                interactive=False,
                label="Segments (optional)",
            )
            funnel_table = gr.Dataframe(
                headers=FUNNEL_COLS,
                col_count=(len(FUNNEL_COLS), "fixed"),
                row_count=(1, "dynamic"),
                interactive=False,
                label="Funnel (if columns exist)",
            )
            json_file = gr.File(label="Download JSON report", interactive=False)

            go = gr.Button("Analyze", variant="primary")
            go.click(
                analyze,
                inputs=[csv_in, boot, seg_on],
                outputs=[memo_md, seg_table, funnel_table, json_file],
                api_name="analyze",
            )

            gr.Markdown(
                "#### CSV columns (required)\n"
                "`user_id`, `variant` (`A`/`B`), `converted` (0/1 or booleans), `revenue` (number)\n\n"
                "#### Optional columns\n"
                "`device`, `country` for segments; `viewed`, `clicked`, `add_to_cart`, `converted` for funnel."
            )

        with gr.Tab("Power"):
            base = gr.Number(value=0.10, label="Baseline conversion rate (0-1)")
            mde = gr.Number(value=0.01, label="Absolute MDE (e.g., 0.01 = +1pp)")
            alpha = gr.Number(value=0.05, label="Alpha (two-sided)")
            pw = gr.Number(value=0.80, label="Desired power")
            out = gr.Markdown("—")
            calc = gr.Button("Compute sample size")
            calc.click(power_calc, inputs=[base, mde, alpha, pw], outputs=[out])

        gr.Markdown(
            "—\nSmall samples ⇒ wide intervals. Bootstrap CIs: resampling users with replacement.\n"
            "SRM check uses χ² on the A/B split.\n"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # Bind to 0.0.0.0 to avoid the 'localhost not accessible' error on some proxies.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=False)
