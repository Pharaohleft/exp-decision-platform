import pandas as pd

REQUIRED_COLS = ["user_id", "variant", "converted", "revenue", "pre_metric"]

def load_experiment_csv(path: str) -> pd.DataFrame:
    # robust read: allow UTF-8 and skip any bad lines
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Normalize and validate
    df["variant"] = df["variant"].astype(str).str.strip().str.upper()
    bad_mask = ~df["variant"].isin(["A", "B"])
    if bad_mask.any():
        bad_vals = df.loc[bad_mask, "variant"].unique().tolist()
        raise ValueError(f"Invalid values in 'variant': {bad_vals}. Use only 'A' or 'B'.")

    df["converted"] = pd.to_numeric(df["converted"], errors="coerce").fillna(0).astype(int)
    if not set(df["converted"].unique()).issubset({0, 1}):
        raise ValueError("Invalid 'converted': must be 0 or 1.")

    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0).astype(float)
    df.loc[df["revenue"] < 0, "revenue"] = 0.0

    df["pre_metric"] = pd.to_numeric(df["pre_metric"], errors="coerce").fillna(0.0).astype(float)

    return df

def summarize_experiment(df: pd.DataFrame) -> dict:
    nA = int((df["variant"] == "A").sum())
    nB = int((df["variant"] == "B").sum())

    pA = float(df.loc[df["variant"] == "A", "converted"].mean()) if nA else float("nan")
    pB = float(df.loc[df["variant"] == "B", "converted"].mean()) if nB else float("nan")

    arpuA = float(df.loc[df["variant"] == "A", "revenue"].mean()) if nA else float("nan")
    arpuB = float(df.loc[df["variant"] == "B", "revenue"].mean()) if nB else float("nan")

    return {
        "rows": int(len(df)),
        "nA": nA,
        "nB": nB,
        "conv_rate_A": pA,
        "conv_rate_B": pB,
        "lift_conv_B_minus_A": (pB - pA) if (pA == pA and pB == pB) else None,
        "arpu_A": arpuA,
        "arpu_B": arpuB,
        "arpu_lift_B_minus_A": (arpuB - arpuA) if (arpuA == arpuA and arpuB == arpuB) else None,
    }
