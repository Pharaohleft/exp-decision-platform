import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
N = 10000

user_id = np.arange(N)
variant = rng.choice(["A","B"], size=N)

# Segments
country = rng.choice(["US","IN","GB","BR"], size=N, p=[0.45, 0.25, 0.15, 0.15])
device  = rng.choice(["web","ios","android"], size=N, p=[0.60, 0.25, 0.15])

# Funnel steps (0/1)
viewed  = (rng.random(N) < 0.70).astype(int)
clicked = (rng.random(N) < (0.25 + 0.02*(variant=="B")) * viewed).astype(int)
add_to_cart = (rng.random(N) < (0.15 + 0.015*(device=="ios")) * clicked).astype(int)

# Conversion depends on variant and country a bit
base_conv = 0.08 + 0.01*(variant=="B") + 0.02*(country=="US")
converted = (rng.random(N) < base_conv * add_to_cart).astype(int)

# Pre-period signal (stronger CUPED)
pre = rng.gamma(1.5, 2.0, size=N)

# Revenue only if converted; depends on pre and variant
rev = converted * (np.maximum(0, rng.normal(8, 3, size=N)) + 0.4*pre + 1.0*(variant=="B"))

df = pd.DataFrame({
    "user_id": user_id,
    "variant": variant,
    "country": country,
    "device": device,
    "viewed": viewed,
    "clicked": clicked,
    "add_to_cart": add_to_cart,
    "converted": converted,
    "revenue": np.round(rev, 2),
    "pre_metric": np.round(pre, 2),
})
df.to_csv(r"data\synth_segmented.csv", index=False)
print("wrote data\\synth_segmented.csv", len(df), "rows")
